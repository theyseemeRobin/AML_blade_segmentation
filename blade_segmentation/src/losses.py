import einops
import torch
from torch.nn import functional as F
from tqdm import tqdm


def val_loss(model, rgbs, gap, num_frames):
    with torch.no_grad():
        mean_motion_loss = 0
        mean_slot_loss = 0
        n_steps = rgbs.shape[1] - (num_frames-1)*gap
        with tqdm(total=n_steps, desc='Validating') as pbar:
            for i in range(0, rgbs.shape[1]):
                indices = torch.arange(i, i+num_frames*gap, gap)
                if indices[-1] >= rgbs.shape[1]:
                    break
                rgb_frames = rgbs[:, indices]
                _, _, motion_mask, slot, _ = model(rgb_frames, training=True)
                slot_loss, motion_loss = compute_total_loss(slot, motion_mask, num_frames)
                pbar.update(1)
                mean_motion_loss = mean_motion_loss - (mean_motion_loss - motion_loss) / (i + 1)
                mean_slot_loss = mean_slot_loss - (mean_slot_loss - slot_loss) / (i + 1)
    return mean_slot_loss, mean_motion_loss


def compute_slot_attention(slot, motion_mask, num_frames, num_points):
    sample = torch.randperm(motion_mask.shape[2]//num_frames)[:num_points]
    attention = einops.rearrange(motion_mask, 'b n (t hw) -> b n t hw', t=num_frames)
    attn_index = torch.argsort(attention, dim=-1, descending=True)
    attn_index = einops.rearrange(attn_index, 'b (t hw) p q -> b t hw p q', t=num_frames)
    attn_index = attn_index[:, :, sample]

    slot_pool = slot[:,None,None,...].repeat([1, num_frames, num_points, 1, 1, 1])
    slot_index = attn_index.unsqueeze(-1).repeat([1, 1, 1, 1, 1, slot.shape[-1]])

    return sample, slot_pool, slot_index


def compute_slot_similarities(slot_pool, slot_index, slot, sample):
    pos = torch.gather(slot_pool, dim=-2, index=slot_index[:, :, :, :, :30])
    neg = torch.gather(slot_pool, dim=-2, index=slot_index[:, :, :, :, -200:])
    query = slot[:, :, sample]

    pos_sim = torch.einsum('bsntkc,bsnc->bsntk', F.normalize(pos, dim=-1), F.normalize(query, dim=-1))
    neg_sim = torch.einsum('bsntkc,bsnc->bsntk', F.normalize(neg, dim=-1), F.normalize(query, dim=-1))

    return F.relu(0.9-pos_sim).mean() + F.relu(neg_sim-0.1).mean()


def compute_motion_attention(attention, motion_mask, attn_index, num_frames, num_points, sample):
    motion_weight = einops.rearrange(attention, 'b (t hw) p q -> b t hw p q', t=num_frames)
    motion_weight = motion_weight[:, torch.arange(num_frames), :, torch.arange(num_frames)]
    motion_weight = motion_weight.transpose(0, 1)
    motion_weight = motion_weight[:, :, sample].detach()

    motion_vector = einops.rearrange(motion_mask, 'b (t n) c -> b t n c', t=num_frames)
    motion_pool = motion_vector.unsqueeze(2).repeat([1, 1, num_points, 1, 1])
    motion_index = attn_index[:, torch.arange(num_frames), :, torch.arange(num_frames)]
    motion_index = motion_index.transpose(0, 1).contiguous()

    return motion_weight, motion_vector, motion_pool, motion_index


def compute_motion_loss(motion_weight, motion_vector, motion_pool, motion_index, sample):
    pos_weight = torch.gather(motion_weight, dim=-1, index=motion_index[:, :, :, :10])
    motion_index_expanded = motion_index.unsqueeze(-1).repeat([1, 1, 1, 1, motion_vector.shape[-1]])

    pos = torch.gather(motion_pool, dim=-2, index=motion_index_expanded[:, :, :, :10])
    neg = torch.gather(motion_pool, dim=-2, index=motion_index_expanded[:, :, :, -50:])
    query = motion_vector[:, :, sample].detach()

    query_score = torch.einsum('btnc,btnc->btn',
                              F.normalize(query, dim=-1, p=1),
                              torch.log(F.normalize(query, dim=-1, p=1)+1e-10))
    query_score = torch.softmax(query_score, dim=-1)

    pos_sim = torch.einsum('btnkc,btnc->btnk',
                          torch.log(F.normalize(pos, dim=-1, p=1)+1e-10),
                          F.normalize(query, dim=-1, p=1))
    neg_sim = torch.einsum('btnkc,btnc->btnk',
                          torch.log(F.normalize(neg, dim=-1, p=1)+1e-10),
                          F.normalize(query, dim=-1, p=1))

    motion_loss = F.relu(neg_sim.unsqueeze(-1)-pos_sim.unsqueeze(-2)+1)
    return torch.einsum('btn,btnpq->btpq', query_score, motion_loss).mean()


def compute_total_loss(slot, motion_mask, num_frames):
    num_points = 8
    sample, slot_pool, slot_index = compute_slot_attention(slot, motion_mask, num_frames, num_points)
    slot_loss = compute_slot_similarities(slot_pool, slot_index, slot, sample)

    attention = einops.rearrange(motion_mask, 'b n (t hw) -> b n t hw', t=num_frames)
    attn_index = torch.argsort(attention, dim=-1, descending=True)
    attn_index = einops.rearrange(attn_index, 'b (t hw) p q -> b t hw p q', t=num_frames)
    attn_index = attn_index[:, :, sample]

    motion_params = compute_motion_attention(attention, motion_mask, attn_index, num_frames, num_points, sample)
    motion_loss = compute_motion_loss(*motion_params, sample)

    return slot_loss, motion_loss
