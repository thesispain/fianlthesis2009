
import pickle
import numpy as np
import sys
import os

DATA_PATH = "data/grouped_flows.pkl"
OUT_PATH = "data/pretrain_50pct_benign.pkl"

def get_label(d):
    lbl = d.get('label', d.get('label_str', 'benign'))
    if isinstance(lbl, int): return lbl
    return 0 if lbl.lower() == 'benign' else 1

print(f"Loading {DATA_PATH}...")
all_data = []
try:
    with open(DATA_PATH, 'rb') as f:
        while True:
            try: all_data.extend(pickle.load(f))
            except EOFError: break
    print(f"Total Flows: {len(all_data)}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Filter Benign
benign = [d for d in all_data if get_label(d) == 0]
print(f"Total Benign Flows: {len(benign)}")

# Shuffle
np.random.seed(42) # Reproducibility
np.random.shuffle(benign)

# Take 50%
subset_size = int(0.5 * len(benign))
subset = benign[:subset_size]
print(f"Selected 50% Subset: {len(subset)} flows")

# Save
print(f"Saving to {OUT_PATH}...")
with open(OUT_PATH, 'wb') as f:
    pickle.dump(subset, f)
print("Done!")
