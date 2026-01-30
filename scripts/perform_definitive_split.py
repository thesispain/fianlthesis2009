
import pickle
import numpy as np
import sys
import os

# CONFIG
MASTER_FILE = "data/unswnb15_full/flows_all.pkl"
OUT_DIR = "data/unswnb15_full"
PRETRAIN_FILE = f"{OUT_DIR}/pretrain_50pct_benign.pkl"
FINETUNE_FILE = f"{OUT_DIR}/finetune_mixed.pkl"
RANDOM_SEED = 42

def main():
    print(f"Loading Master Dataset from {MASTER_FILE}...")
    try:
        with open(MASTER_FILE, 'rb') as f:
            all_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {MASTER_FILE} not found.")
        sys.exit(1)
        
    print(f"Total Flows: {len(all_data)}")
    
    # 1. Separate Benign and Attack
    benign = []
    attack = []
    
    for d in all_data:
        if d['label'] == 0:
            benign.append(d)
        else:
            attack.append(d)
            
    print(f"--- Separation ---")
    print(f"Benign: {len(benign)}")
    print(f"Attack: {len(attack)}")
    
    # 2. Shuffle Benign
    print(f"Shuffling Benign flows (Seed {RANDOM_SEED})...")
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(benign)
    
    # 3. Split Benign 50/50
    midpoint = len(benign) // 2
    benign_pretrain = benign[:midpoint]
    benign_finetune = benign[midpoint:]
    
    print(f"--- Split Benign ---")
    print(f"Set A (Pretrain): {len(benign_pretrain)}")
    print(f"Set B (Finetune): {len(benign_finetune)}")
    
    # 4. Construct Datasets
    # Pretrain Set = Benign A
    # Finetune Set = Benign B + All Attacks
    finetune_mixed = benign_finetune + attack
    # Shuffle Finetune Mixed just to mix attacks in
    np.random.shuffle(finetune_mixed)
    
    print(f"--- Finalizing ---")
    print(f"Pretrain Set: {len(benign_pretrain)} (Benign Only)")
    print(f"Finetune Set: {len(finetune_mixed)} (Mixed)")
    
    # 5. Save
    print(f"Saving Pretrain Set to {PRETRAIN_FILE}...")
    with open(PRETRAIN_FILE, 'wb') as f:
        pickle.dump(benign_pretrain, f)
        
    print(f"Saving Finetune Set to {FINETUNE_FILE}...")
    with open(FINETUNE_FILE, 'wb') as f:
        pickle.dump(finetune_mixed, f)
        
    print("Done. Splits are perfectly disjoint.")

if __name__ == "__main__":
    main()
