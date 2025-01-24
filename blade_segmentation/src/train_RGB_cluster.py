import os
import time
import einops
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import SequentialSampler

import wandb
import gc
from argparse import ArgumentParser
import random

from src.utils import save_on_master, Augment_GPU_pre
from tqdm import tqdm
import src.utils as ut
import src.config as cg
from src.model.model_cluster import AttEncoder
from src.eval_oneshot import eval

# Set matmul precision to medium
torch.set_float32_matmul_precision('medium')


def train_rgb_cluster(args):
    ut.init_distributed_mode(args)
    torch.autograd.set_detect_anomaly(True)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + ut.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    print(f"GPU number:{torch.cuda.device_count()}")
    print(f"world_size:{ut.get_world_size()}")
    lr = args.lr
    batch_size = args.batch_size
    resume_path = args.resume_path
    num_t = args.num_t
    dino_path = args.dino_path
    num_frames = args.num_frames
    grad_iter = args.grad_iter
    args.resolution = (192, 384)
    aug_gpu = Augment_GPU_pre(args)
    
    # setup log and model path, initialize tensorboard,
    logPath, modelPath, resultsPath = cg.setup_path(args)
    print(logPath)

    trn_dataset, val_dataset, resolution, _ = cg.setup_dataset(args)
    
    if True:  # args.distributed:
        num_tasks = ut.get_world_size()
        global_rank = ut.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            trn_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

    wandb.init(
        project='aml-blade-segmentation',
        config=args,
        mode='online' if args.wandb_online else 'offline'
    )

    trn_loader = ut.FastDataLoader(
        trn_dataset,
        sampler=sampler_train,
        num_workers=8,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False, # We have too few samples to drop any
        multiprocessing_context=None                   # NOTE (Robin): "fork" did not work so I set it to default (None)
    )
    
    val_loader = ut.FastDataLoader(
        val_dataset, 
        num_workers=1, 
        batch_size=1, # For stability 
        shuffle=False, 
        pin_memory=True, 
        drop_last=False
    )
        
    model = AttEncoder(resolution=resolution, num_t=num_t, num_frames=num_frames, dino_path=dino_path)
    
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    it = 0
    
    def get_params_groups(model):
        encoder_reg = []
        encoder_noreg = []
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if 'encoder' in name:
                if name.endswith(".bias") or len(param.shape) == 1:
                    encoder_noreg.append(param)
                else:
                    encoder_reg.append(param)
            else:
                if name.endswith(".bias") or len(param.shape) == 1:
                    not_regularized.append(param)
                else:
                    regularized.append(param)
        return [{'params': encoder_reg}, {'params': encoder_noreg, 'weight_decay': 0.},
            {'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
    
    param_groups = get_params_groups(model)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    optimizer.param_groups[0]['weight_decay'] = 0.04

    def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule
    
    lr_scheduler = cosine_scheduler(lr, 1e-6, args.num_epochs, len(trn_loader), 0)
    wd_scheduler = cosine_scheduler(0.1, 0.1, args.num_epochs, len(trn_loader))
    
    if resume_path:
        print('resuming from checkpoint')
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        del checkpoint

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # NOTE (Robin): model did not have a `module` attribute so I changed `model.module.encoder...` to `model.encoder...`
    #               did this for the other model.module calls too.
    for name, p in model.encoder.named_parameters():
        p.requires_grad = False

    print('======> start training {}, {}, use {}.'.format(args.dataset, args.verbose, device))
    grad_step = 1
    timestart = time.time()
    iter_per_epoch = int(10000 // (num_tasks * args.batch_size))
    
    for epoch in tqdm(range(args.num_epochs), desc='epochs'):
        
        losses = {
            'total_loss': 0,
            'slot_loss': 0,
            'motion_loss': 0,
        }
        
        if args.distributed:
            trn_loader.sampler.set_epoch(it//iter_per_epoch)
        for sample in trn_loader:
            for i, param_group in enumerate(optimizer.param_groups):
                if i < 2:
                    weight = 0.0 if it < grad_iter else 0.1
                else:
                    weight = 1.0
                param_group["lr"] = lr_scheduler[it] * weight
                if i % 2 == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_scheduler[it]

            model.train()
            model.encoder.eval()
            rgb = sample
            rgb = rgb.float().to(device)
            
            _, _, motion_mask, slot, _ = model(aug_gpu(rgb), training=True)

            slot_loss, motion_loss = compute_total_loss(slot, motion_mask, num_frames)
            loss = (slot_loss + motion_loss) / grad_step 
            
            loss.backward()
            
            losses['total_loss'] += loss.detach().cpu().numpy()
            losses['slot_loss'] += slot_loss.detach().cpu().numpy()
            losses['motion_loss'] += motion_loss.detach().cpu().numpy()

            if (it+1) % grad_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            it += 1
            
        # Normalize the losses
        for key in losses:
            losses[key] /= len(trn_loader)

        if (epoch % args.log_freq == 0 or epoch == args.num_epochs - 1):
            # print('epoch {},'.format(epoch),
            #     'time {:.01f}s,'.format(time.time() - timestart),
            #     'learning rate {:.05f}'.format(lr_scheduler[it]),
            #     'slot loss {:.05f}'.format(slot_loss),
            #     'motion loss {:.05f}.'.format(float(motion_loss)),
            #     'total loss {:.05f}.'.format(float(loss.detach().cpu().numpy())))
            
            timestart = time.time()
            
            # Log in wandb
            wandb.log(
                {'train/total_loss': losses['total_loss'],
                 'train/slot_loss': losses['slot_loss'],
                 'train/motion_loss': losses['motion_loss'],
                 # it-1 because the scheduler is updated at the end of the iteration, so `it-1` matches the lr used
                 'learning_rate': lr_scheduler[it - 1]},
                step=epoch)

        if epoch % args.save_freq == 0 and it > 0:
            filename = os.path.join(modelPath, 'checkpoint_{}.pth'.format(it))
            save_on_master({
                'epoch': it,
                'model_state_dict': model_without_ddp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, filename)
        
        # Evaluate the model, logs results to wandb
        if (epoch % args.eval_freq == 0 and epoch > 0) or epoch == args.num_epochs - 1:   
            
            # Free memory before evaluation
            torch.cuda.empty_cache()
            
            # Delete out of scope variables
            del slot, motion_mask, rgb, slot_loss, motion_loss
            
            # Garbage collection
            gc.collect()
            
            eval(val_loader, model, device, args, save_path=resultsPath, train=True)
    
    # Save the final model
    filename = os.path.join(modelPath, 'checkpoint_final.pth')
    save_on_master({
        'iteration': it,
        'model_state_dict': model_without_ddp.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, filename)

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


def train_rgb_cluster_parse_args():
    parser = ArgumentParser()
    #optimization
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--num_train_steps', type=int, default=6e4) #60k
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--loss_scale', type=float, default=100)
    parser.add_argument('--ent_scale', type=float, default=1.0)
    parser.add_argument('--cons_scale', type=float, default=1.0)
    parser.add_argument('--grad_iter', type=int, default=0)
    #settings
    parser.add_argument('--dataset', type=str, default='DAVIS')
    parser.add_argument('--with_rgb', action='store_true')
    parser.add_argument('--num_frames', type=int, default=3)
    parser.add_argument('--num_t', type=int, default=1)

    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--verbose', type=str, default=None)
    parser.add_argument('--basepath', type=str, default="/home/user/work/")
    parser.add_argument('--output_path', type=str, default="/home/user/work/shuangrui/TEST_log/test")
    parser.add_argument('--dino_path', type=str, default="/home/user/work/dino_deitsmall16_pretrain.pth")
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # NOTE (Robin): The following arguments are used, but were not added to the original parser, so I added them myself
    parser.add_argument('--gap', type=int, default=2, help='the sampling stride of frames')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = train_rgb_cluster_parse_args()
    args.inference = False
    args.distributed = True
    train_rgb_cluster(args)
