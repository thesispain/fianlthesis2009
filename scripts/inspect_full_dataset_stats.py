
import pickle
import sys
from collections import Counter

FILE = "data/unswnb15_full/flows_all.pkl"

def main():
    print(f"Loading {FILE}...")
    try:
        with open(FILE, 'rb') as f:
            data = pickle.load(f)
            
        total = len(data)
        print(f"Total Flows: {total}")
        
        # Count labels
        # Structure is dict: {'label': int, 'label_str': str ...}
        
        benign = 0
        attack = 0
        
        # Sample check
        if total > 0:
            print(f"Sample Entry Keys: {data[0].keys()}")
            print(f"Sample features shape: {data[0]['features'].shape}")
            
        for d in data:
            if d['label'] == 0:
                benign += 1
            else:
                attack += 1
                
        print("\n--- STATISTICS ---")
        print(f"Total:  {total}")
        print(f"Benign: {benign} ({benign/total*100:.1f}%)")
        print(f"Attack: {attack} ({attack/total*100:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
