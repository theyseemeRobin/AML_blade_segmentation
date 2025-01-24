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
import faiss.contrib.torch_utils
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

def mem_efficient_inference(masks_collection, rgbs, gts, model, T, ratio, tau, device):
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
    
    # Replace hierarchical_cluster with k-means clustering
    dist = faiss_kmeans_cluster(q, k, scale, num_clusters=8, device=device) # Use FAISS GPU K-means 
    
    dist = einops.rearrange(dist, '(s p) (t h w) -> t s p h w', t=T, p=1, h=H)
    mask = dist.unsqueeze(1)
    for i in range(T):
        masks_collection[i].append(mask[i])
    t3 = time.time()
    print('Attention map, clustering time:', t2-t1, t3-t2)
    print('Total time:', t3-t1)
    return masks_collection

def faiss_kmeans_cluster(q, k_in, scale, num_clusters=8, device='cuda', chunk_size=1):
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
    kmeans = faiss.Kmeans(d=d, k=num_clusters, niter=10, verbose=False, gpu=False)
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

def eval(val_loader, model, device, ratio, tau, save_path=None, writer=None, train=False):
    with torch.no_grad():
        t = time.time()
        model.eval()
        mean = torch.tensor([0.43216, 0.394666, 0.37645])
        std = torch.tensor([0.22803, 0.22145, 0.216989])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        aug_list = VideoSequential(
            normalize_video,
            data_format="BTCHW",
            same_on_frame=True)
        print(' --> running inference')
        for val_sample in tqdm(val_loader, desc='evaluating video sequences'):
            rgbs, gts, category, val_idx = val_sample
            print(f'-----------process {category} sequence in one shot-------')
            rgbs = rgbs.float().to(device)  # b t c h w
            rgbs = aug_list(rgbs)
            gts = gts.float().to(device)  # b t c h w
            T = rgbs.shape[1]
            masks_collection = {}
            for i in range(T):
                masks_collection[i] = []
            masks_collection = mem_efficient_inference(masks_collection, rgbs, gts, model, T, ratio, tau, device)
            torch.save(masks_collection, save_path+'/%s.pth' % category[0])

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
