import os
import time
import einops
import sys
import cv2
import numpy as np
import gc

import src.utils as ut
import src.config as cg
from src.model.model_cluster import AttEncoder
import PIL.Image as Image

import torch

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch.nn.functional as F
import kornia
from kornia.augmentation.container import VideoSequential
from kornia.enhance import Denormalize, Normalize

from vos_benchmark.benchmark import benchmark
from tqdm import tqdm

from src.cluster import mem_efficient_hierarchical_cluster, kmeans_cluster, sklearn_dbscan_cluster

def mem_efficient_inference(masks_collection, rgbs, gts, model, T, args, device):
    
    ratio = args.ratio
    
    bs = 1
    feats = []
    ## extract frame-wise dino features
    t1 = time.time()
    for i in tqdm(range(0, T, bs), desc='extracting features from frames'):
        input = rgbs[:, i:i+bs]
        input = einops.rearrange(input, 'b t c h w -> (b t) c h w')
        with torch.no_grad():
            _, _, _, feat = model.encoder(input)
            feats.append(feat.cpu())
    feats = torch.cat(feats, 0).to(device) # t c h w
    print('spatio-temporal feature:', feats.shape)

    ## calculate the spatio-temporal attention, use sparse sampling on keys to reduce computational cost
    T, C, H, W = feats.shape
    num_heads = model.temporal_transformer[0].attn.num_heads
    feats = einops.rearrange(feats, 't c h w -> t (h w) c')
    feats = model.temporal_transformer[0].norm1(feats) # t hw c
    qkv = model.temporal_transformer[0].attn.qkv(feats)
    qkv = qkv.reshape(T, H*W, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4) # 3 t h hw c
    q, k, _ = qkv.unbind(0) # t h hw c # Originally put on CPU
    key_indices = torch.arange(T//ratio) * ratio # sparse sampling the keys with a sparsity ratio
    k = k[key_indices]
    scale = model.temporal_transformer[0].attn.scale

    ## clustering on the spatio-temporal attention maps and produce segmentation for the whole video
    t2 = time.time()
    
    # Delete unnecessary variables
    del feats, qkv
    
    gc.collect()
    
    if args.clustering_algorithm == 'kmeans':
        dist = kmeans_cluster(q, k, scale, args, device=device)
    elif args.clustering_algorithm == 'dbscan':
        dist = sklearn_dbscan_cluster(q, k, scale, args, device=device)
    elif args.clustering_algorithm == 'hierarchical':
        dist = mem_efficient_hierarchical_cluster(q, k, scale, args, device=device)
    else:
        raise ValueError(f"Unknown clustering algorithm: {args.clustering_algorithm}")
    
    print('Distance matrix:', dist.shape)
    dist = einops.rearrange(dist, '(s p) (t h w) -> t s p h w', t=T, p=1, h=H)
    mask = dist.unsqueeze(1)
    for i in range(T):
        masks_collection[i].append(mask[i])
    t3 = time.time()
    print('Attention map, clustering time:', t2-t1, t3-t2)
    print('Total time:', t3-t1)
    return masks_collection

def eval(val_loader, model, device, args, save_path=None, writer=None, train=False):
    
    with torch.no_grad():
        t = time.time()
        model.eval()
        mean = torch.tensor([0.43216, 0.394666, 0.37645])
        std = torch.tensor([0.22803, 0.22145, 0.216989])
        
        # Create video-appropriate normalization/denormalization
        normalize_video = VideoSequential(
            Normalize(mean, std),
            data_format="BTCHW",
            same_on_frame=True
        )
        
        denormalize_video = VideoSequential(
            Denormalize(mean.to(device), std.to(device)),
            data_format="BTCHW",
            same_on_frame=True
        )
        
        print(' --> running inference')
        for val_sample in tqdm(val_loader, desc='evaluating video sequences'):
            rgbs, gts, category, val_idx = val_sample
            print(f'-----------process {category} sequence in one shot-------')
            rgbs = rgbs.float().to(device)  # b t c h w
            
            # Normalize through video-sequential
            rgbs = normalize_video(rgbs)
            gts = gts.float().to(device)  # b t c h w
            T = rgbs.shape[1]
            print("Number of frames: ", T)
            masks_collection = {}
            for i in range(T):
                masks_collection[i] = []
            masks_collection = mem_efficient_inference(masks_collection, rgbs, gts, model, T, args, device)
            torch.save(masks_collection, save_path+'/%s.pth' % category[0])
            
            # Prepare original frames
            original_frames = denormalize_video(rgbs).cpu()
            
            # Resolution of windmill format
            original_resolution = (args.gt_x, args.gt_y)
            save_davis_masks(masks_collection, category, save_path, original_resolution, T)
            
            # Create segmentation video
            if args.save_video:
                create_segmentation_video(original_frames, masks_collection, category, save_path, T)
    
    gt_subdir = 'DAVIS_Masks' # For the test set
    if train:
        gt_subdir = 'val/DAVIS_Masks'
        gt_dir = os.path.join(args.basepath, gt_subdir)
        
    pred_dir = os.path.join(save_path, 'Annotations') # Might need to change this during testing
    
    results = benchmark([gt_dir], [pred_dir])

    # J - Jaccard (Region similarity)
    # JF - Jaccard (Boundary similarity)
    # F - F-measure
    J, JF, F = results[:3]
    
    # Lists are returned since benchmark() can handle multiple datasets
    return J[0], JF[0], F[0]
            
def load_davis_palette(palette_path):
    """Load DAVIS palette from text file"""
    palette = np.loadtxt(palette_path, dtype=np.uint8)
    
    # Verify palette format
    if palette.shape != (256, 3):
        raise ValueError(f"Invalid palette shape {palette.shape}, expected (256, 3)")
    
    # Flatten to 768-length list (R1, G1, B1, R2, G2, B2, ...)
    return palette.flatten()

def save_davis_masks(masks_collection, category, save_path, original_resolution, T):
    """
    Save masks in DAVIS-compatible indexed PNG format
    Each Pixel in the png contains 1 channel with the object ID that starts from 1
    """
    # Load palette
    davis_palette = load_davis_palette('config_dir/palette.txt')
    
    # Create output directory
    mask_dir = os.path.join(save_path, 'Annotations', category[0])
    os.makedirs(mask_dir, exist_ok=True)
    
    # Get original dimensions
    h, w = original_resolution
    
    # Process all frames explicitly
    for frame_idx in range(T):  # 0-based indexing up to T-1
        if frame_idx not in masks_collection or not masks_collection[frame_idx]:
            # Create empty mask
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            # Get mask tensor [num_objects, mask_h, mask_w]
            mask_tensor = masks_collection[frame_idx][0].squeeze().cpu().numpy()
            
            # 1. Convert to object IDs using mask dimensions
            object_ids = np.argmax(mask_tensor, axis=0).astype(np.uint8) + 1
            
            # 2. Upscale to original resolution
            mask = cv2.resize(object_ids, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 3. Ensure background is 0 and clip values
            mask = np.clip(mask, 0, 254)

        # Save with palette
        img = Image.fromarray(mask, mode='P')
        img.putpalette(davis_palette)
        frame_path = os.path.join(mask_dir, f"{frame_idx:05d}.png")
        img.save(frame_path, format='PNG', transparency=254, compress_level=0)
    
    print(f'Saved DAVIS masks to {mask_dir}')
            
def create_segmentation_video(original_frames, masks_collection, category, save_path, T):
    """
    Creates a video visualization of segmentation masks overlaid on original frames.
    
    Args:
        original_frames (torch.Tensor): Denormalized video frames
        masks_collection (list): Collection of segmentation masks
        category (list): Category name for the video
        save_path (str): Base directory to save the video
        T (int): Number of frames
    """
    # Create video writer
    video_dir = os.path.join(save_path, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f'{category[0]}_segmentation.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # Adjust based on your frame rate
    
    # Get frame dimensions from first mask
    first_mask = masks_collection[0][0].cpu()
    
    # Directly extract spatial dimensions from last two axes
    mask_h, mask_w = first_mask.shape[-2:]  # Works for any number of dimensions

    h, w = original_frames.shape[-2:]  # Original frame dimensions

    # Handle potential resolution mismatch
    scale_factor_h = h // mask_h
    scale_factor_w = w // mask_w

    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for i in range(T):
        # Process original frame
        frame = original_frames[0, i].numpy()
        frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Process masks
        if len(masks_collection[i]) == 0:
            out.write(frame)
            continue
        
        masks = masks_collection[i][0].cpu().squeeze()  # Remove batch dimension
        num_masks = masks.shape[0]
        
        # Create segmentation map
        seg_map = torch.argmax(masks, dim=0).numpy().astype(np.uint8)
        
        # Upscale to original resolution
        if (scale_factor_h > 1) or (scale_factor_w > 1):
            seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Apply color map
        color_map = cv2.applyColorMap((seg_map * (255 // max(num_masks, 1))).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay on original frame
        overlay = cv2.addWeighted(frame, 0.7, color_map, 0.3, 0)
        
        # Write to video
        out.write(overlay)

    out.release()
    print(f'Saved segmentation video to {video_path}')

def main(args):
    epsilon = 1e-5
    batch_size = args.batch_size 
    resume_path = args.resume_path
    attn_drop_t = args.attn_drop_t
    path_drop = args.path_drop
    num_t = args.num_t
    args.resolution = tuple(args.resolution)

    # setup log and model path, initialize tensorboard,
    # initialize dataloader
    trn_dataset, val_dataset, resolution, in_out_channels = cg.setup_dataset(args)
    val_loader = ut.FastDataLoader(
        val_dataset, num_workers=8, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('======> start inference {}, {}, use {}.'.format(args.dataset, args.verbose, device))

    model = AttEncoder(resolution=resolution,
                        path_drop=path_drop,
                        attn_drop_t=attn_drop_t,
                        num_t=num_t)
    model.to(device)

    it = 0
    if resume_path:
        print('resuming from checkpoint')
        checkpoint = torch.load(resume_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        it = checkpoint['iteration']
        loss = checkpoint['loss']
        model.eval()
    else:
        print('no checkpouint found')
        sys.exit(0)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    eval(val_loader, model, device, args.ratio, args.tau, save_path=args.save_path, train=False)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    #optimization
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--num_train_steps', type=int, default=3e5) #30k
    #data
    parser.add_argument('--dataset', type=str, default='DAVIS2017')
    parser.add_argument('--resolution', nargs='+', type=int)
    #architecture
    parser.add_argument('--num_frames', type=int, default=3)
    parser.add_argument('--path_drop', type=float, default=0.2)
    parser.add_argument('--attn_drop_t', type=float, default=0.4)
    parser.add_argument('--num_t', type=int, default=1)
    parser.add_argument('--gap', type=int, default=2, help='the sampling stride of frames')
    parser.add_argument('--ratio', type=int, default=10, help='key frame sampling rate in inference')
    parser.add_argument('--tau', type=float, default=1.0, help='distance threshold in clustering')
    #misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--verbose', type=str, default=None)
    parser.add_argument('--basepath', type=str, default="/path/to/DAVIS-2017")
    parser.add_argument('--output_path', type=str, default="./OUTPUT/")
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    args = parser.parse_args()
    main(args)
