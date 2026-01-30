
import pickle
import numpy as np
import sys
import torch

DATA_PATH = "data/unswnb15_full/pretrain_50pct_benign.pkl"

def main():
    print(f"Loading {DATA_PATH}...")
    try:
        with open(DATA_PATH, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Loaded {len(data)} flows")
    
    # Extract features
    # Each row is dict with 'features' key -> (32, 5)
    # 0: Proto, 1: LogLen, 2: Flags, 3: LogIAT, 4: Dir
    
    # Take a sample of 1000
    sample_size = min(len(data), 10000)
    print(f"Sampling {sample_size} for analysis...")
    sample = data[:sample_size]
    
    features = np.stack([d['features'] for d in sample]) # (N, 32, 5)
    
    print(f"Shape: {features.shape}")
    
    # Check variance per feature
    # Reshape to (N*32, 5) to check global distribution
    flat = features.reshape(-1, 5)
    
    names = ['Proto', 'LogLen', 'Flags', 'LogIAT', 'Dir']
    for i, name in enumerate(names):
        col = flat[:, i]
        print(f"\nFeature: {name}")
        print(f"  Mean: {np.mean(col):.4f}")
        print(f"  Std:  {np.std(col):.4f}")
        print(f"  Min:  {np.min(col):.4f}")
        print(f"  Max:  {np.max(col):.4f}")
        print(f"  Unique Vals (approx): {len(np.unique(col[:1000]))}")

if __name__ == "__main__":
    main()
