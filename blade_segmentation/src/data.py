import os
import cv2
import glob
import torch
import random
import einops
import numpy as np
import glob as gb
from torch.utils.data import Dataset

def readRGB(sample_dir, resolution):
    """Load and process RGB images with memory optimization"""
    rgb = cv2.imread(sample_dir, cv2.IMREAD_COLOR)
    try:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error processing {sample_dir}: {str(e)}")
        raise
    
    # Resize with preserved aspect ratio
    if resolution[0] == -1:
        h = (rgb.shape[0] // 8) * 8
        w = (rgb.shape[1] // 8) * 8
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        rgb = cv2.resize(rgb, (resolution[1], resolution[0]), interpolation=cv2.INTER_LANCZOS4)
    
    # Rearrange dimensions and keep as uint8
    return einops.rearrange(rgb, 'h w c -> c h w').astype(np.uint8)

    
class Dataloader(Dataset):
    def __init__(self, data_dir, resolution, dataset, seq_length=3, gap=4, train=True, val_seq=None):
        self.dataset = dataset
        self.eval = eval
        self.data_dir = data_dir
        self.img_dir = data_dir[1]
        self.gap = gap
        self.resolution = resolution
        self.seq_length = seq_length
        if train:
            self.train = train
            # self.img_dir = '/path/to/ytvis/train'
            self.seq = list([os.path.basename(x) for x in gb.glob(os.path.join(self.img_dir, '*'))])
        else:
            self.train = train
            self.seq = val_seq

    def __len__(self):
        # if self.train:
        #     return 10000 # What?
        # else:
            return len(self.seq)

    def __getitem__(self, idx):
        if self.train:
            seq_name = random.choice(self.seq)
            seq = os.path.join(self.img_dir, seq_name, '*.jpg')
            imgs = gb.glob(seq)
            imgs.sort()
            length = len(imgs)
            gap = self.gap
            while gap*(self.seq_length//2) >= length-gap*(self.seq_length//2)-1:
                gap = gap-1
            ind = random.randint(gap*(self.seq_length//2), length-gap*(self.seq_length//2)-1)
            
            seq_ids = [ind+gap*(i-(self.seq_length//2)) for i in range(self.seq_length)]

            rgb_dirs = [imgs[i] for i in seq_ids]
            rgbs = [readRGB(rgb_dir, self.resolution) for rgb_dir in rgb_dirs]
            out_rgb = np.stack(rgbs, 0) ## T, C, H, W 
            return out_rgb

        else:
            if self.dataset == 'FBMS':
                seq_name = self.seq[idx]
                rgb_dirs = sorted(os.listdir(os.path.join(self.data_dir[1], seq_name)))
                rgb_dirs = [os.path.join(self.data_dir[1], seq_name, x) for x in rgb_dirs if x.endswith(".jpg")]
                rgbs = np.stack([readRGB(rgb_dir, self.resolution) for rgb_dir in rgb_dirs], axis=0)
                gt_dirs = os.listdir(os.path.join(self.data_dir[2], seq_name))
                gt_dirs = sorted([gt for gt in gt_dirs if gt.endswith(".png")])
                val_idx = [int(x[:-4])-int(gt_dirs[0][:-4]) for x in gt_dirs if x.endswith(".png")]
                gt_dirs = [os.path.join(self.data_dir[2], seq_name, x) for x in gt_dirs if x.endswith(".png")]  
                return rgbs, seq_name, val_idx
            elif self.dataset == 'turbines_O':
                seq_name = self.seq[idx]
                tot = len(glob.glob(os.path.join(self.data_dir[1], seq_name, '*')))
                rgb_dirs = [os.path.join(self.data_dir[1], seq_name, f'{seq_name.zfill(2)}_{str(i).zfill(5)}.png') for i in range(tot-1)]
                gt_dirs = [os.path.join(self.data_dir[2], seq_name, f'{seq_name.zfill(2)}_{str(i).zfill(5)}.png') for i in range(tot-1)]
                rgbs = np.stack([readRGB(rgb_dir, self.resolution) for rgb_dir in rgb_dirs], axis=0)
                return rgbs, seq_name, [i for i in range(tot-1)]
            else:
                seq_name = self.seq[idx]
                tot = len(glob.glob(os.path.join(self.data_dir[1], seq_name, '*')))
                rgb_dirs = [os.path.join(self.data_dir[1], seq_name, str(i).zfill(5)+'.jpg') for i in range(tot-1)]
                gt_dirs = [os.path.join(self.data_dir[2], seq_name, str(i).zfill(5)+'.png') for i in range(tot-1)]
                rgbs = np.stack([readRGB(rgb_dir, self.resolution) for rgb_dir in rgb_dirs], axis=0)
                return rgbs, seq_name, [i for i in range(tot-1)]
                
