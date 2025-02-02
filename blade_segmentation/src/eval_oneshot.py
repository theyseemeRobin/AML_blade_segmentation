import os
import time
import einops
import sys
import cv2
import numpy as np
import gc
import wandb

import src.utils as ut
import src.config as cg
from src.model.model_cluster import AttEncoder
from src.losses import val_loss
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

from src.cluster import mem_efficient_hierarchical_cluster, kmeans_cluster, spectral_cluster

def mem_efficient_inference(masks_collection, rgbs, model, T, args, device):
    
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
    elif args.clustering_algorithm == 'spectral':
        dist = spectral_cluster(q, k, scale, args, device=device)
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
    
    # Frames per second
    fps = T / (t3 - t1)
    return masks_collection, fps

def bgs_inference(masks_collection, rgbs, T, args, device):
    """Background subtraction based inference using pre-processed frames"""
    t1 = time.time()
    
    # Process each frame
    for i in range(T):
        # Extract frame and convert to uint8
        frame = rgbs[0, i].cpu().numpy()  # Remove batch dim
        frame = np.transpose(frame, (1, 2, 0))
        frame = (frame * 255).astype(np.uint8)
        
        # Create binary mask (assuming frame is already background subtracted)
        binary_mask = (frame.mean(axis=2) > 128).astype(np.uint8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        if num_labels > 1:  # If we found any components
            # Find and remove background (largest component)
            unique_ids, counts = np.unique(labels, return_counts=True)
            background_id = unique_ids[np.argmax(counts)]
            
            # Create new labels without background
            new_labels = np.zeros_like(labels)
            current_new_id = 1
            for old_id in unique_ids:
                if old_id != background_id:
                    new_labels[labels == old_id] = current_new_id
                    current_new_id += 1
            
            # Convert to one-hot encoding
            num_objects = len(unique_ids) - 1  # Subtract background
            h, w = new_labels.shape
            one_hot = np.zeros((num_objects + 1, h, w))
            for j in range(num_objects + 1):
                one_hot[j] = (new_labels == j)
            
            # Convert to tensor and add to collection
            mask_tensor = torch.from_numpy(one_hot).float().unsqueeze(0)
            masks_collection[i].append(mask_tensor.to(device))
        else:
            # If no components found, add empty mask
            h, w = binary_mask.shape
            empty_mask = torch.zeros((1, 1, h, w)).to(device)
            masks_collection[i].append(empty_mask)
    
    t2 = time.time()
    fps = T / (t2 - t1)
    
    return masks_collection, fps

def eval(val_loader, model, device, args, save_path=None, writer=None, train=False, skip_loss=False):
    
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
        mean_slot_loss = 0
        mean_motion_loss = 0
        
        fps_list = []
        for sample_idx, val_sample in enumerate(tqdm(val_loader, desc='evaluating video sequences')):
            rgbs, category, val_idx = val_sample
            print(f'-----------process {category} sequence in one shot-------')
            rgbs = rgbs.float().to(device)  # b t c h w
            
            # Normalize through video-sequential
            rgbs = normalize_video(rgbs)
            T = rgbs.shape[1]
            print("Number of frames: ", T)
            masks_collection = {}
            for i in range(T):
                masks_collection[i] = []
            
            slot_loss, motion_loss = 0, 0
            if not skip_loss:
                slot_loss, motion_loss = val_loss(model, rgbs, num_frames=args.num_frames, gap=args.gap)
                
            mean_slot_loss = mean_slot_loss - (mean_slot_loss - slot_loss) / (sample_idx + 1)
            mean_motion_loss = mean_motion_loss - (mean_motion_loss - motion_loss) / (sample_idx + 1)
            
            if args.clustering_algorithm:
                
                if args.clustering_algorithm != 'bgs':
                    masks_collection, fps = mem_efficient_inference(masks_collection, rgbs, model, T, args, device)
                else:
                    masks_collection, fps = bgs_inference(masks_collection, rgbs, T, args, device)
                
                fps_list.append(fps)
                
                # Prepare original frames
                original_frames = denormalize_video(rgbs).cpu()
                
                # Resolution of windmill format
                original_resolution = (args.gt_x, args.gt_y)
                save_davis_masks(masks_collection, category, save_path, original_resolution, T)
                
                # Create segmentation video
                if args.save_video:
                    create_segmentation_video(original_frames, masks_collection, category, save_path, T)

            if args.save_attention_slice:
                create_attention_figure(model, rgbs, args.gap, T, args.n_attention_slices)
    
    gt_subdir = 'DAVIS_Masks' # For the test set
    if train:
        gt_subdir = 'val/DAVIS_Masks'
        
    gt_dir = os.path.join(args.basepath, gt_subdir)
    
    pred_dir = os.path.join(save_path, 'Annotations') # Might need to change this during testing
    
    avg_fps = sum(fps_list) / len(fps_list)
    J, JF, F = 0, 0, 0
    if args.clustering_algorithm:
        results = benchmark([gt_dir], [pred_dir])

        # J - Jaccard (Region similarity)
        # JF - Mean?
        # F - Boundary F-measure
        J, JF, F = results[:3]
        J, JF, F = J[0], JF[0], F[0]
    
    # Lists are returned since benchmark() can handle multiple datasets
    return J, JF, F, mean_slot_loss, mean_motion_loss, avg_fps
            
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
    Background (largest mask) is removed and set to 0
    """
    davis_palette = load_davis_palette('config_dir/palette.txt')
    mask_dir = os.path.join(save_path, 'Annotations', category[0])
    os.makedirs(mask_dir, exist_ok=True)
    
    h, w = original_resolution
    
    for frame_idx in range(T):
        if frame_idx not in masks_collection or not masks_collection[frame_idx]:
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask_tensor = masks_collection[frame_idx][0].squeeze().cpu().numpy()
            object_ids = np.argmax(mask_tensor, axis=0).astype(np.uint8) + 1
            
            # Find the background (largest component)
            unique_ids, counts = np.unique(object_ids, return_counts=True)
            if len(unique_ids) > 1:  # If we have more than just background
                background_id = unique_ids[np.argmax(counts)]
                
                # Set background to 0 and shift other IDs to maintain consecutive numbering
                new_ids = np.zeros_like(object_ids)
                current_new_id = 1
                for old_id in unique_ids:
                    if old_id != background_id:
                        new_ids[object_ids == old_id] = current_new_id
                        current_new_id += 1
                
                object_ids = new_ids
            
            # Upscale to original resolution
            mask = cv2.resize(object_ids, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = np.clip(mask, 0, 254)

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
        frame = np.transpose(frame, (1, 2, 0))
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Process masks
        if len(masks_collection[i]) == 0:
            out.write(frame)
            continue
        
        masks = masks_collection[i][0].cpu().squeeze()
        num_masks = masks.shape[0]
        
        # Create initial segmentation map
        seg_map = torch.argmax(masks, dim=0).numpy().astype(np.uint8)
        
        # Find and remove background (largest component)
        unique_ids, counts = np.unique(seg_map, return_counts=True)
        if len(unique_ids) > 1:
            background_id = unique_ids[np.argmax(counts)]
            
            # Create new segmentation map without background
            new_seg_map = np.zeros_like(seg_map)
            current_new_id = 1
            for old_id in unique_ids:
                if old_id != background_id:
                    new_seg_map[seg_map == old_id] = current_new_id
                    current_new_id += 1
            
            seg_map = new_seg_map
        
        # Upscale to original resolution
        if (scale_factor_h > 1) or (scale_factor_w > 1):
            seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Apply color map (adjusted for number of non-background masks)
        actual_num_masks = len(np.unique(seg_map)) - 1  # subtract 1 to exclude 0
        if actual_num_masks > 0:
            color_map = cv2.applyColorMap((seg_map * (255 // max(actual_num_masks, 1))).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.4, color_map, 0.3, 0)
        else:
            overlay = frame
        
        out.write(overlay)

    out.release()
    print(f'Saved segmentation video to {video_path}')


def create_attention_figure(model, rgbs, gap, num_frames, num_figures=1):
    with torch.no_grad():
        n_steps = rgbs.shape[1] - (num_frames-1)*gap
        with tqdm(total=n_steps, desc='Validating') as pbar:
            for i in range(0, min(num_figures, rgbs.shape[1])):
                indices = torch.arange(i, i+num_frames*gap, gap)
                if indices[-1] >= rgbs.shape[1]:
                    break
                rgb_frames = rgbs[:, indices]
                _, _, attention_map, _, _ = model(rgb_frames, training=True)
                # reshape attention mask to (THW,THW)
                # Select a 'query' point in the spatio-temporal by finding the maximum value in the attention mask
                # to obtain a THW attention map from the query to all
                # other points in the spatio-temporal domain
                # for each step in the temporal dimension T get an image of the HW slice.
                # add a dot indicating the location of the prompt, and save all T attention maps.



def train_clusterer(args):
    batch_size = args.batch_size 
    resume_path = args.resume_path
    attn_drop_t = args.attn_drop_t
    path_drop = args.path_drop
    num_t = args.num_t
    args.resolution = (args.rgb_x, args.rgb_y)
    logPath, modelPath, resultsPath = cg.setup_path(args)

    # setup log and model path, initialize tensorboard,
    # initialize dataloader
    trn_dataset, val_dataset, resolution, in_out_channels = cg.setup_dataset(args)
    val_loader = ut.FastDataLoader(
        val_dataset, num_workers=1, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('======> start inference {}, using {}.'.format(args.dataset, device))

    model = AttEncoder(
        resolution=resolution,
        path_drop=path_drop,
        attn_drop_t=attn_drop_t,
        num_t=num_t,
        device=device,
        num_frames=args.num_frames
    )
    
    model.to(device)

    it = 0
    if resume_path:
        print('resuming from checkpoint')
        checkpoint = torch.load(resume_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        it = checkpoint['iteration']
        loss = checkpoint['loss']
        model.eval()
    else:
        print('no checkpoint found')
        sys.exit(0)

    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)

    print(args.train)
    J, JF, F, _, _, fps = eval(val_loader, model, device, args, save_path=resultsPath, train=args.train, skip_loss=True)
    wandb.init(project=args.wandb_project, config=args)
    
    # Log metrics
    wandb.log({
        'Region Similarity': J,
        'Mean': JF,
        'Boundary F Measure': F,
        'Frames per Second': fps
    })
    
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
    train_clusterer(args)
