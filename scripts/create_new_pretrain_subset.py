
import pickle
import numpy as np
import sys
import os

# CONFIG
INPUT_FILE = "data/unswnb15_full/flows_all.pkl"
OUT_FILE = "data/unswnb15_full/pretrain_50pct_benign.pkl"
RANDOM_SEED = 42
PERCENTAGE = 0.50

def main():
    print(f"Loading Master Dataset from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'rb') as f:
            all_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please run generate_full_unsw_dataset.py first.")
        sys.exit(1)
        
    print(f"Total Flows: {len(all_data)}")
    
    # 1. Filter Benign
    print("Filtering for Benign flows (Label=0)...")
    benign_data = [d for d in all_data if d['label'] == 0]
    n_benign = len(benign_data)
    print(f"Total Benign Flows: {n_benign} ({n_benign/len(all_data)*100:.1f}%)")
    
    # 2. Shuffle
    print(f"Shuffling with Seed {RANDOM_SEED}...")
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(benign_data)
    
    # 3. Take 50%
    target_count = int(n_benign * PERCENTAGE)
    subset = benign_data[:target_count]
    print(f"Selected {target_count} flows ({PERCENTAGE*100}%) for Pretraining.")
    
    # 4. Save
    print(f"Saving to {OUT_FILE}...")
    with open(OUT_FILE, 'wb') as f:
        pickle.dump(subset, f)
        
    print("Done. Success.")

if __name__ == "__main__":
    main()
