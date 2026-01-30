
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import random

class PretrainDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading data from {data_path}...")
        try:
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded {len(self.data)} flows.")
        except FileNotFoundError:
            print(f"Error: {data_path} not found.")
            self.data = []

        # Convert to Tensor List to save memory during training? 
        # Or keep as dicts. We'll process on the fly to save RAM if dataset is 1.6M.
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        
        # Extract features if dict, or assume tensor input
        if isinstance(row, dict):
            # [Proto, Len, Flags, IAT, Dir]
            f = row['features']
        else:
            f = row
            
        # Ensure correct shape (32, 5)
        # Assuming data is already preprocessed to (32, 5) by generating script
        # But if not, we can pad here. 
        # The new generator ensures (32, 5).
        
        x = torch.from_numpy(f).float()
        
        # AUGMENTATION (CutMix)
        # To do efficient CutMix, we need a random "other flow".
        # We can't access random other items easily in __getitem__ without passing the whole list.
        # Strategy: Return index, and let Collate function handle mixing?
        # Or just pick a random index here (if self.data is accessible).
        
        other_idx = random.randint(0, len(self.data)-1)
        other_row = self.data[other_idx]
        other_f = other_row['features']
        
        x_aug = self.apply_cutmix(f, other_f)
        x_aug = torch.from_numpy(x_aug).float()
        
        return x, x_aug
        
    def apply_cutmix(self, original, noise, ratio=0.4):
        """Replaces a chunk of 'original' with 'noise'"""
        L = original.shape[0]
        patch_len = int(L * ratio)
        if patch_len == 0: patch_len = 1
        
        # Random position
        start = random.randint(0, L - patch_len)
        
        mixed = original.copy()
        mixed[start:start+patch_len] = noise[start:start+patch_len]
        
        return mixed
