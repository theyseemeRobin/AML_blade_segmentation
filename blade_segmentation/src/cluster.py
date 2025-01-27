import torch
import einops
from tqdm import tqdm
import faiss
import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering

def kl_distance(final_set, attn):
    ## kl divergence between two distributions
    self_entropy = - torch.einsum('nc,nc->n', final_set, torch.log(final_set)).unsqueeze(-1) - torch.einsum('mc,mc->m', attn, torch.log(attn)).unsqueeze(0)
    cross_entropy = - torch.einsum('nc,mc->nm', final_set, torch.log(attn)) - torch.einsum('mc,nc->nm', attn, torch.log(final_set))
    distance = cross_entropy - self_entropy
    return distance

def chunk_generator(q, k, scale, chunk_size, description=None):
    """Memory-efficient chunk generator with head-wise processing"""
    hw = q.shape[2]  # Spatial dimension (height*width)
    num_heads = q.shape[1]
    total_chunks = (q.shape[0] + chunk_size - 1) // chunk_size
    
    with tqdm(total=total_chunks, desc=description) as pbar:
        for t in range(0, q.shape[0], chunk_size):
            q_chunk = q[t:t+chunk_size]
            chunk_sample = None
            
            # Process each attention head separately
            for h in range(num_heads):
                # Extract head-specific queries and keys
                q_head = q_chunk[:, h]  # [T, n, c]
                k_head = k[:, h]        # [S, m, c]
                
                # Compute head-specific attention
                head_attn = torch.einsum('tnc,smc->tsnm', q_head, k_head) * scale
                head_attn = einops.rearrange(head_attn, 't s n m -> (t n) (s m)')
                head_attn = head_attn.softmax(dim=-1)
                
                # Accumulate results and clean up
                if chunk_sample is None:
                    chunk_sample = head_attn
                else:
                    chunk_sample += head_attn
                del head_attn, q_head, k_head
            
            # Finalize sample for this chunk
            chunk_sample.div_(num_heads)
            indices = torch.arange(
                t * hw,
                (t + q_chunk.shape[0]) * hw,
                device=q.device
            )
            
            yield chunk_sample, indices
            pbar.update(1)

def mem_efficient_hierarchical_cluster(q, k, scale, args, device):
    tau = args.tau
    num_iter = args.n_iter
    ts = args.chunk_size
    bs = 10000  # Batch size for distance computation
    final_set = []

    # First hierarchy clustering using generator
    gen = chunk_generator(q, k, scale, ts, "First hierarchy")
    for sample, _ in gen:
        sample = sample.to(device)
        distance = kl_distance(sample, sample)
        
        keep_set = []
        for i in range(0, distance.shape[0], bs):
            indices = (distance[i:i+bs] <= tau)
            dist = torch.einsum('bn,nc->bc', indices.float(), sample)
            dist = dist / dist.sum(dim=1, keepdim=True)
            keep_set.append(dist)
        
        keep_set = torch.cat(keep_set, dim=0)
        distance_chunk = kl_distance(keep_set, keep_set)
        indicator = torch.ones(len(keep_set), device=device)
        
        for i in range(len(keep_set)):
            if indicator[i] == 0:
                continue
            indices = (distance_chunk[i] <= tau) * (indicator > 0)
            dist = torch.mean(keep_set[indices], dim=0)
            final_set.append(dist)
            indicator[indices] = 0
            
    keep_set = final_set

    # Subsequent hierarchies
    for t in range(num_iter):
        final_set = []
        keep_set = torch.stack(keep_set, dim=0)
        distance = kl_distance(keep_set, keep_set)
        indicator = torch.ones(len(keep_set)).to(device)
        for i in range(len(keep_set)):
            if indicator[i] == 0:
                continue
            indices = (distance[i] <= tau) * (indicator > 0)
            dist = torch.mean(keep_set[indices], dim=0)
            final_set.append(dist)
            indicator[indices] = 0
        if len(final_set) == len(keep_set):
            break
        keep_set = final_set
    final_set = torch.stack(final_set, dim=0)

    # Final mask assignment using generator
    final_mask = torch.zeros(final_set.shape[0], q.shape[0]*q.shape[2]).to(device)
    gen = chunk_generator(q, k, scale, ts, "Final assignments")
    for sample, indices in gen:
        sample = sample.to(device)
        distance = kl_distance(final_set.to(device), sample)
        nms_set = torch.argmin(distance, dim=0)
        for i in range(final_mask.shape[0]):
            final_mask[i, indices[nms_set==i]] = 1

    print('cluster centroids:', final_set.shape)    
    return final_mask

def kmeans_cluster(q, k, scale, args, device='cuda'):
    num_clusters = args.num_clusters
    ts = args.chunk_size
    final_centroids = []

    # Initial Clustering with k-means (using generator)
    gen = chunk_generator(q, k, scale, ts, "Initial Clustering")
    for sample, _ in gen:
        sample = sample.cpu().numpy().astype('float32')

        # Apply k-means
        kmeans = faiss.Kmeans(sample.shape[1], num_clusters, niter=args.n_iter, gpu=False)
        kmeans.train(sample)
        final_centroids.append(kmeans.centroids)

    # Convert cluster centroids to a tensor, stacking along a new dimension
    final_centroids = np.stack(final_centroids, axis=0)  # Shape: (num_chunks, num_clusters, feature_dim)
    
    # Reshape to 2D for FAISS: (num_chunks * num_clusters, feature_dim)
    final_centroids = final_centroids.reshape(-1, final_centroids.shape[-1])

    # Refine Centroids with k-means
    kmeans = faiss.Kmeans(final_centroids.shape[1], num_clusters, niter=args.n_iter, gpu=False)
    kmeans.train(final_centroids)
    final_centroids = torch.tensor(kmeans.centroids, dtype=torch.float32, device=device)

    # Final Mask Assignment (using generator)
    final_mask = torch.zeros(num_clusters, q.shape[0] * q.shape[2]).to(device)
    gen = chunk_generator(q, k, scale, ts, "Final Mask Assignment")
    for sample, indices in gen:
        sample = sample.to(device)

        # Calculate distances to centroids
        distances = torch.cdist(sample, final_centroids)

        # Assign to closest cluster
        assignments = torch.argmin(distances, dim=1)

        # Fill the mask
        for i in range(num_clusters):
            final_mask[i, indices[assignments == i]] = 1

    print('Cluster centroids:', final_centroids.shape)
    return final_mask

def spectral_cluster(q, k, scale, args, device='cuda'):
    
    num_clusters = args.num_clusters
    ts = args.chunk_size
    chunk_features = []
    all_indices = []
    all_assignments = []
    
    # Initial clustering phase
    gen = chunk_generator(q, k, scale, ts, "Initial Spectral Clustering")
    for sample, indices in gen:
        sample = sample.cpu().numpy().astype('float32')
        
        spectral = SpectralClustering(
            n_clusters=num_clusters,
            affinity='rbf',
        )
        chunk_labels = spectral.fit_predict(sample)
        
        # Store cluster means and assignments
        for i in range(num_clusters):
            mask = chunk_labels == i
            if np.any(mask):
                cluster_mean = sample[mask].mean(axis=0)
                chunk_features.append(cluster_mean)
        
        all_indices.append(indices)
        all_assignments.append(chunk_labels)
    
    # Refinement phase
    chunk_features = np.array(chunk_features)
    spectral_refine = SpectralClustering(
        n_clusters=num_clusters,
        affinity='rbf',
    )
    refined_labels = spectral_refine.fit_predict(chunk_features)
    
    # Create final mask
    final_mask = torch.zeros(num_clusters, q.shape[0] * q.shape[2]).to(device)
    feature_idx = 0
    
    for chunk_labels, indices in zip(all_assignments, all_indices):
        for i in range(num_clusters):
            mask = chunk_labels == i
            if np.any(mask):
                cluster_id = refined_labels[feature_idx]
                final_mask[cluster_id, indices[mask]] = 1
                feature_idx += 1
    
    print('Spectral clustering complete')
    return final_mask

def sklearn_dbscan_cluster(q, k, scale, args, device='cuda'):
    final_mask = torch.zeros(0, q.shape[0]*q.shape[2], device=device)
    
    gen = chunk_generator(q, k, scale, args.chunk_size, "DBSCAN clustering")
    for sample, indices in gen:
        db = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(sample.cpu().numpy())
        labels = db.labels_
        valid_labels = labels[labels != -1]
        
        if len(valid_labels) == 0:
            continue
            
        # Create chunk-specific mask
        unique_labels = np.unique(valid_labels)
        chunk_mask = torch.zeros(len(unique_labels), final_mask.shape[1], device=device)
        
        for i, label in enumerate(unique_labels):
            chunk_mask[i, indices[labels == label]] = 1
            
        final_mask = torch.cat([final_mask, chunk_mask], dim=0)
        
    return final_mask