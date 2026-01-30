
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

from config import Config
try:
    from model import BiMambaEncoder
    # Test if we can instantiate it (will fail if no GPU logic)
    # Actually model.py imports mamba_ssm which might fail at import time or runtime
except:
    pass
from mamba_cpu import BiMambaCPU

def load_data_subset(path, count=10000, label_filter=None):
    """Load a subset of flows. If label_filter is set (0 or 1), filter by it."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    if label_filter is not None:
        filtered = [d for d in data if d['label'] == label_filter]
        # Shuffle
        np.random.shuffle(filtered)
        return filtered[:count]
    else:
        np.random.shuffle(data)
        return data[:count]

def get_embeddings(model, flow_list):
    """Compute embeddings for a list of flows (dicts)"""
    # Convert to tensor
    features = np.stack([d['features'] for d in flow_list])
    x = torch.from_numpy(features).float().to(Config.DEVICE)
    
    # Batch processing to avoid OOM
    batch_size = 512
    embeddings = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i+batch_size]
            z, _ = model(batch)
            embeddings.append(z.cpu())
            
    return torch.cat(embeddings, dim=0)

def main():
    print("🚀 Starting Anomaly Detection Evaluation (Cosine Method)")
    
    # 1. Load Model
    if torch.cuda.is_available():
         model = BiMambaEncoder(d_model=Config.D_MODEL).to(Config.DEVICE)
         print("Using GPU Mamba (Fast)")
    else:
         model = BiMambaCPU(d_model=Config.D_MODEL).to(Config.DEVICE)
         print("⚠️ GPU Missing: Using BiMambaCPU (Slow but Functional)")
    # Load Weights
    weight_path = f"{Config.CHECKPOINT_DIR}/epoch_10.pth"
    # Fallback if 10 doesn't exist?
    try:
        model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))
        print(f"Loaded weights from {weight_path}")
    except:
        print(f"Could not load {weight_path}, trying latest.pth")
        model.load_state_dict(torch.load(f"{Config.CHECKPOINT_DIR}/latest.pth", map_location=Config.DEVICE))

    # 2. Prepare Data
    # Reference Set (Benign from Training)
    print("Loading Reference Set (Benign)...")
    data_ref = load_data_subset(Config.DATA_PATH, count=10000) # 10k reference flows form Phase 1
    
    # Query Set (Benign from Test) -> We use Finetune Set Benign
    print("Loading Eval Benign...")
    data_benign_eval = load_data_subset("data/unswnb15_full/finetune_mixed.pkl", count=5000, label_filter=0)
    
    # Query Set (Attack from Test)
    print("Loading Eval Attack...")
    data_attack_eval = load_data_subset("data/unswnb15_full/finetune_mixed.pkl", count=5000, label_filter=1)
    
    print(f"Reference Size: {len(data_ref)}")
    print(f"Eval Benign Size: {len(data_benign_eval)}")
    print(f"Eval Attack Size: {len(data_attack_eval)}")
    
    # 3. Compute Embeddings
    print("Computing Embeddings...")
    emb_ref = get_embeddings(model, data_ref)        # (10000, 128)
    emb_benign = get_embeddings(model, data_benign_eval) # (5000, 128)
    emb_attack = get_embeddings(model, data_attack_eval) # (5000, 128)
    
    # Normalize for Cosine Similarity
    emb_ref = F.normalize(emb_ref, dim=1).to(Config.DEVICE)
    emb_benign = F.normalize(emb_benign, dim=1).to(Config.DEVICE)
    emb_attack = F.normalize(emb_attack, dim=1).to(Config.DEVICE)
    
    # 4. Calculate Scores (Max Cosine Sim)
    # Score = max_j ( eval_i . ref_j )
    
    def compute_max_sim(eval_emb, ref_emb, chunk_size=100):
        # Result: list of max scores
        max_scores = []
        # Process in chunks to save GPU memory on Matrix Mul
        for i in tqdm(range(0, len(eval_emb), chunk_size), desc="Computing Sim"):
            batch = eval_emb[i:i+chunk_size] # (B, 128)
            # Sim Matrix: (B, 128) @ (10000, 128).T -> (B, 10000)
            sim_matrix = torch.mm(batch, ref_emb.T)
            # Max over dim 1
            vals, _ = torch.max(sim_matrix, dim=1)
            max_scores.extend(vals.cpu().numpy())
        return np.array(max_scores)

    print("Scoring Benign Eval...")
    scores_benign = compute_max_sim(emb_benign, emb_ref)
    
    print("Scoring Attack Eval...")
    scores_attack = compute_max_sim(emb_attack, emb_ref)
    
    # 5. Analysis
    # Benign should be HIGH (close to 1.0)
    # Attack should be LOW (far from reference)
    
    print("\n--- RESULTS ---")
    print(f"Avg MaxSim (Benign): {np.mean(scores_benign):.4f} +/- {np.std(scores_benign):.4f}")
    print(f"Avg MaxSim (Attack): {np.mean(scores_attack):.4f} +/- {np.std(scores_attack):.4f}")
    
    # Metrics
    y_true = [0]*len(scores_benign) + [1]*len(scores_attack) # 0=Benign, 1=Attack 
    # BUT wait, Anomaly Detection means "Attack is Anomaly".
    # Usually: Anomaly Score = Distance.
    # Here: Anomaly Score = Similarity.
    # So High Score = Benign (Normal). Low Score = Attack (Anomaly).
    # To use standard AUC (higher is better for positive class), we should invert score or labels.
    # Let's align: Pos Class (1) is Benign. Neg Class (0) is Attack.
    # Then AUC represents ability to detect Benign.
    # Or flip: Score = 1 - Sim.
    
    y_scores = np.concatenate([scores_benign, scores_attack])
    # Let's interpret Score as "Benign Probability".
    # Target: Benign=1, Attack=0
    y_target = np.array([1]*len(scores_benign) + [0]*len(scores_attack))
    
    auc = roc_auc_score(y_target, y_scores)
    print(f"\nAUROC (Detection Capability): {auc:.4f}")
    print("(1.0 = Perfect Separation, 0.5 = Random)")
    
    # Save Plot?
    # Histogram
    # print hist text
    print("\nDistribution (Text Hist):")
    bins = np.linspace(0, 1, 11)
    
    hist_b, _ = np.histogram(scores_benign, bins)
    hist_a, _ = np.histogram(scores_attack, bins)
    
    print(f"{'Bin':<10} | {'Benign':<10} | {'Attack':<10}")
    print("-" * 36)
    for i in range(len(bins)-1):
        print(f"{bins[i]:.1f}-{bins[i+1]:.1f}   | {hist_b[i]:<10} | {hist_a[i]:<10}")

if __name__ == "__main__":
    main()
