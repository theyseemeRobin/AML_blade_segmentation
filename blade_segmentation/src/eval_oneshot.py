import os
import time
import einops
import sys
import cv2
import numpy as np

import src.utils as ut
import src.config as cg
from src.model.model_cluster import AttEncoder

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch.nn.functional as F
import kornia
from kornia.augmentation.container import VideoSequential
from kornia.enhance import Denormalize, Normalize
import faiss.contrib.torch_utils
from sklearn.cluster import DBSCAN
from tqdm import tqdm

def kl_distance(final_set, attn):
    ## kl divergence between two distributions
    self_entropy = - torch.einsum('nc,nc->n', final_set, torch.log(final_set)).unsqueeze(-1) - torch.einsum('mc,mc->m', attn, torch.log(attn)).unsqueeze(0)
    cross_entropy = - torch.einsum('nc,mc->nm', final_set, torch.log(attn)) - torch.einsum('mc,nc->nm', attn, torch.log(final_set))
    distance = cross_entropy - self_entropy
    return distance

def hierarchical_cluster(attn, tau, num_iter, device):
    # attn t hw c
    ts = 3
    bs = 10000
    attn = attn / attn.sum(dim=-1, keepdim=True)
    final_set = []

    ## use a temporal window for clustering of the first hierarchy to speed up clustering process
    for t in range(0, attn.shape[0], ts):
        sample = attn[t:t+ts].view(-1, attn.shape[-1]).to(device)
        distance = kl_distance(sample, sample)
        keep_set = []
        for i in range(0, distance.shape[0], bs):
            indices = (distance[i:i+bs] <= tau) # bs hw
            dist = torch.einsum('bn,nc->bc', indices.float(), sample) # bs hw
            dist = dist / dist.sum(dim=1, keepdim=True)
            keep_set.append(dist)
        keep_set = torch.cat(keep_set, dim=0)
        distance = kl_distance(keep_set, keep_set)
        indicator = torch.ones(len(keep_set)).to(device)
        for i in range(len(keep_set)):
            if indicator[i] == 0:
                continue
            indices = (distance[i] <= tau) * (indicator > 0) # K
            dist = torch.mean(keep_set[indices], dim=0)
            final_set.append(dist)
            indicator[indices] = 0
    keep_set = final_set

    ## clustering on all frames at the following hierarchies until no change
    for t in range(num_iter):
        final_set = []
        keep_set = torch.stack(keep_set, dim=0) # K hw
        distance = kl_distance(keep_set, keep_set)
        indicator = torch.ones(len(keep_set)).to(device) # K
        for i in range(len(keep_set)):
            if indicator[i] == 0:
                continue
            indices = (distance[i] <= tau) * (indicator > 0) # K
            dist = torch.mean(keep_set[indices], dim=0)
            final_set.append(dist)
            indicator[indices] = 0
        if len(final_set) == len(keep_set):
            break
        keep_set = final_set
    final_set = torch.stack(final_set, dim=0)

    ## calculate cluster assignments as object segmentation masks
    distance = kl_distance(final_set.to(attn.device), attn.view(-1, attn.shape[-1]))
    nms_set = torch.argmin(distance, dim=0)
    final_mask = torch.zeros(final_set.shape[0], attn.shape[0]*attn.shape[1]).to(attn.device)
    print('cluster centroids:', final_set.shape)
    for i in range(final_mask.shape[0]):
        final_mask[i, nms_set==i] = 1
    return final_mask

def mem_efficient_hierarchical_cluster(q, k, scale, tau, num_iter, device):
    # q t head hw c
    # k tk head hw c
    ts = 3 # temporal window size, num_frames?
    bs = 10000
    final_set = []

    ## use a temporal window for clustering of the first hierarchy to speed up clustering process
    for t in tqdm(range(0, q.shape[0], ts), desc='clustering on the first hierarchy'):
        attn = torch.einsum('qhnc,khmc->qkhnm', q[t:t+ts], k) * scale
        attn = einops.rearrange(attn, 'q k h n m -> (q n) h (k m)')
        attn = attn.softmax(dim=-1)
        sample = attn.mean(dim=1).to(device) # thw khw
        distance = kl_distance(sample, sample)
        keep_set = []
        for i in range(0, distance.shape[0], bs):
            indices = (distance[i:i+bs] <= tau) # bs hw
            dist = torch.einsum('bn,nc->bc', indices.float(), sample) # bs hw
            dist = dist / dist.sum(dim=1, keepdim=True)
            keep_set.append(dist)
        keep_set = torch.cat(keep_set, dim=0)
        distance = kl_distance(keep_set, keep_set)
        indicator = torch.ones(len(keep_set)).to(device)
        for i in range(len(keep_set)):
            if indicator[i] == 0:
                continue
            indices = (distance[i] <= tau) * (indicator > 0) # K
            dist = torch.mean(keep_set[indices], dim=0)
            final_set.append(dist)
            indicator[indices] = 0
    keep_set = final_set

    ## clustering on all frames at the following hierarchies until no change
    for t in tqdm(range(num_iter), desc='clustering on the following hierarchies'):
        final_set = []
        keep_set = torch.stack(keep_set, dim=0) # K hw
        distance = kl_distance(keep_set, keep_set)
        indicator = torch.ones(len(keep_set)).to(device) # K
        for i in range(len(keep_set)):
            if indicator[i] == 0:
                continue
            indices = (distance[i] <= tau) * (indicator > 0) # K
            dist = torch.mean(keep_set[indices], dim=0)
            final_set.append(dist)
            indicator[indices] = 0
        if len(final_set) == len(keep_set):
            break
        keep_set = final_set
    final_set = torch.stack(final_set, dim=0)

    ## calculate cluster assignments as object segmentation masks
    final_mask = torch.zeros(final_set.shape[0], q.shape[0]*q.shape[2]).to(q.device)
    for t in tqdm(range(0, q.shape[0], ts), desc='calculating cluster assignments'):
        attn = torch.einsum('qhnc,khmc->qkhnm', q[t:t+ts], k) * scale
        attn = einops.rearrange(attn, 'q k h n m -> (q n) h (k m)')
        attn = attn.softmax(dim=-1)
        sample = attn.mean(dim=1).to(device) # thw khw
        distance = kl_distance(final_set.to(sample.device), sample) # K thw
        nms_set = torch.argmin(distance, dim=0)
        for i in range(final_mask.shape[0]):
            final_mask[:, t*q.shape[2]:(t+ts)*q.shape[2]][i, nms_set==i] = 1
    print('cluster centroids:', final_set.shape)           
    return final_mask

# This apparently takes 300 GB of RAM
def inference(masks_collection, rgbs, gts, model, T, ratio, tau, device):
    bs = 1
    feats = []
    ## extract frame-wise dino features
    for i in tqdm(range(0, T, bs), desc='extracting features from frames'):
        input = rgbs[:, i:i+bs]
        input = einops.rearrange(input, 'b t c h w -> (b t) c h w')
        with torch.no_grad():
            _, _, _, feat = model.encoder(input)
            feats.append(feat.cpu())
    feats = torch.cat(feats, 0).to(device) # t c h w

    ## calculate the spatio-temporal attention, use sparse sampling on keys to reduce computational cost
    T, C, H, W = feats.shape
    num_heads = model.temporal_transformer[0].attn.num_heads
    feats = einops.rearrange(feats, 't c h w -> t (h w) c')
    feats = model.temporal_transformer[0].norm1(feats) # t hw c
    qkv = model.temporal_transformer[0].attn.qkv(feats)
    qkv = qkv.reshape(T, H*W, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4) # 3 t h hw c
    q, k, _ = qkv.cpu().unbind(0) # t h hw c
    key_indices = torch.arange(T//ratio) * ratio # sparse sampling the keys with a sparsity ratio
    k = k[key_indices]
    attention = torch.einsum('qhnc,khmc->qkhnm', q, k) * model.temporal_transformer[0].attn.scale
    attention = einops.rearrange(attention, 'q k h n m -> (q n) h (k m)')
    attention = attention.softmax(dim=-1)
    attention = attention.mean(dim=1) # thw khw
    print('spatio-temporal attention matrix:', attention.shape)

    ## clustering on the spatio-temporal attention maps and produce segmentation for the whole video
    dist = hierarchical_cluster(attention.view(T, H*W, -1), tau=tau, num_iter=10000, device=device)
    dist = einops.rearrange(dist, '(s p) (t h w) -> t s p h w', t=T, p=1, h=H)
    mask = dist.unsqueeze(1)
    for i in range(T):
        masks_collection[i].append(mask[i])
    return masks_collection

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
    
    if args.clustering_algorithm == 'kmeans':
        dist = faiss_kmeans_cluster(q, k, scale, args, device=device)
    elif args.clustering_algorithm == 'dbscan':
        dist = sklearn_dbscan_cluster(q, k, scale, args, device=device)
    else:
        raise ValueError(f"Unknown clustering algorithm: {args.clustering_algorithm}")
    
    dist = einops.rearrange(dist, '(s p) (t h w) -> t s p h w', t=T, p=1, h=H)
    mask = dist.unsqueeze(1)
    for i in range(T):
        masks_collection[i].append(mask[i])
    t3 = time.time()
    print('Attention map, clustering time:', t2-t1, t3-t2)
    print('Total time:', t3-t1)
    return masks_collection

def faiss_kmeans_cluster(q, k_in, scale, args, device='cuda'):
    
    num_clusters = args.num_clusters
    chunk_size = args.chunk_size
    
    batch_size = q.shape[0]
    seq_len = q.shape[2]
    total_len = batch_size * seq_len
    final_mask = torch.zeros(num_clusters, total_len, device=q.device)

    # Process first chunk to get dimensions
    q_chunk = q[0:1]
    attn = torch.einsum('qhnc,khmc->qkhnm', q_chunk, k_in) * scale
    attn = einops.rearrange(attn, 'q k h n m -> (q n) h (k m)')
    attn = attn.softmax(dim=-1)
    sample = attn.mean(dim=1)
    d = sample.shape[1]
    
    # Initialize FAISS kmeans once
    kmeans = faiss.Kmeans(d=d, k=num_clusters, niter=200, verbose=False, gpu=False)
    kmeans.train(sample.cpu().numpy().astype('float32'))
    
    # Process chunks
    for t in tqdm(range(0, batch_size, chunk_size)):
        # Process chunk
        q_chunk = q[t:t+chunk_size]
        attn = torch.einsum('qhnc,khmc->qkhnm', q_chunk, k_in) * scale
        attn = einops.rearrange(attn, 'q k h n m -> (q n) h (k m)')
        attn = attn.softmax(dim=-1)
        sample = attn.mean(dim=1)
        
        # Get cluster assignments for chunk
        sample_np = sample.cpu().numpy().astype('float32')
        _, chunk_labels = kmeans.index.search(sample_np, 1)
        
        # Update mask for this chunk
        start_idx = t * seq_len
        end_idx = min((t + chunk_size) * seq_len, total_len)
        chunk_labels_torch = torch.from_numpy(chunk_labels.reshape(-1)).to(q.device)
        chunk_indices = torch.arange(start_idx, end_idx, device=q.device)
        final_mask[chunk_labels_torch, chunk_indices] = 1
        
        # Clean up
        torch.cuda.empty_cache()
        del q_chunk, attn, sample, sample_np, chunk_labels

    return final_mask

def sklearn_dbscan_cluster(q, k_in, scale, args, device='cuda'):
    eps = args.eps
    min_samples = args.min_samples
    chunk_size = args.chunk_size

    batch_size = q.shape[0]
    seq_len = q.shape[2]
    total_len = batch_size * seq_len
    final_mask = torch.zeros(0, total_len, device=device)

    for t in tqdm(range(0, batch_size, chunk_size), desc='DBSCAN clustering'):
        q_chunk = q[t:t+chunk_size]
        attn = torch.einsum('qhnc,khmc->qkhnm', q_chunk, k_in) * scale
        attn = einops.rearrange(attn, 'q k h n m -> (q n) h (k m)')
        attn = attn.softmax(dim=-1)
        sample = attn.mean(dim=1).cpu().numpy()

        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(sample)
        labels = db.labels_

        # Create binary masks for each cluster
        unique_labels = np.unique(labels[labels != -1])  # Exclude noise
        if len(unique_labels) == 0:
            continue
            
        # Initialize chunk mask
        chunk_mask = torch.zeros(len(unique_labels), total_len, device=device)
        
        # Calculate indices for this chunk
        start_idx = t * seq_len
        end_idx = min((t + chunk_size) * seq_len, total_len)
        chunk_indices = torch.arange(start_idx, end_idx, device=device)
        
        # Assign labels to mask
        for i, label in enumerate(unique_labels):
            mask_indices = torch.tensor(np.where(labels == label)[0], device=device)
            valid_indices = mask_indices[mask_indices < len(chunk_indices)]
            chunk_mask[i, chunk_indices[valid_indices]] = 1

        final_mask = torch.cat([final_mask, chunk_mask], dim=0)

    return final_mask

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
            masks_collection = {}
            for i in range(T):
                masks_collection[i] = []
            masks_collection = mem_efficient_inference(masks_collection, rgbs, gts, model, T, args, device)
            torch.save(masks_collection, save_path+'/%s.pth' % category[0])
            
            # Prepare original frames
            original_frames = denormalize_video(rgbs).cpu()
            # Create segmentation video
            create_segmentation_video(original_frames, masks_collection, category, save_path, T)
            
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
