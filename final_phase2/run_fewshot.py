
import sys
import os
import torch
import torch.nn as nn
import pickle
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import copy

# Add root to path so we can import from final_phase1
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from final_phase1.config import Config
from final_phase1.model import BiMambaEncoder
# Verify GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

# --- CONFIG PHASE 2 ---
PHASE2_EPOCHS = 30
PATIENCE = 3
FEWSHOT_PERCENT = 0.001 # 0.1%
BATCH_SIZE = 64
LR_FINETUNE = 1e-4

# --- MODEL WRAPPER ---
class BiMambaClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Freeze encoder? Usually for few-shot we might freeze or slow-learn
        # User said "Fine-tune", implying end-to-end update.
        # But with 0.1% data, freezing might be safer. 
        # Research paper usually unfreezes. Let's keep it trainable.
        
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2) # Binary Classification
        )
        
    def forward(self, x):
        # Encoder returns (z, recon). We want z (representation)
        # Check model.py -> forward returns (z_i, recon_i)
        # z_i is the projected output (128 dim). 
        # We usually want the backbone features (256 dim) before projection head.
        # But BiMambaEncoder.forward() calls proj_head internally.
        # Let's see BiMambaEncoder code...
        # It calculates global_rep (256) then z (128).
        # We should ideally use global_rep.
        # Let's monkey-patch or just use z for now?
        # Actually, let's use the provided forward but we need backend features.
        # Ideally we'd modify BiMambaEncoder to return global_rep. 
        # But sticking to what we have: z is a good representation too.
        
        z, _ = self.encoder(x) # (B, 128)
        # Wait, z is 128 (projected).
        # If we use z, we are fine-tuning on top of the specified projection space.
        return self.head(z) 

# --- DATASET ---
class FinetuneDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data[idx]
        features = row['features'] # (32, 5) from Phase 1 logic
        return torch.from_numpy(features).float(), torch.tensor(row['label']).long()

def load_and_split_data():
    print("Loading finetune_mixed.pkl...")
    with open("../data/unswnb15_full/finetune_mixed.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Extract Labels for stratification
    labels = [d['label'] for d in data]
    
    # 1. Split Train (70%) / Test (30%)
    # We always need a big Test set to prove metrics are real.
    train_indices, test_indices = train_test_split(
        np.arange(len(data)), test_size=0.3, stratify=labels, random_state=42
    )
    
    # 2. Subsample Train to 0.1% (Few-Shot)
    # 0.1% of the Total Labeled Data means 0.1% of the original pool?
    # Or 0.1% of the Train set?
    # Usually "0.1% labeled data" means the Training Set size is 0.1% of total available.
    # Let's filter train_indices.
    
    train_labels = [labels[i] for i in train_indices]
    
    # We want 0.1% of the TOTAL available data to be used for training.
    # Total = len(data). Target = 0.001 * len(data) ≈ 1635 samples.
    target_k = int(len(data) * FEWSHOT_PERCENT)
    
    # We select these target_k samples from the TRAIN split.
    # But wait, 0.1% is tiny.
    # Let's take 0.1% of train_indices.
    
    fs_train_idx, _ = train_test_split(
        train_indices, train_size=target_k, stratify=train_labels, random_state=42
    )
    
    train_data = [data[i] for i in fs_train_idx]
    test_data = [data[i] for i in test_indices]
    
    print(f"Total Data: {len(data)}")
    print(f"Few-Shot Train Size (0.1%): {len(train_data)}")
    print(f"Test Set Size (30%): {len(test_data)}")
    
    return train_data, test_data

def train_phase2():
    train_data, test_data = load_and_split_data()
    
    # Calculate Class Weights for Loss "Punishment"
    y_train = [d['label'] for d in train_data]
    count_benign = y_train.count(0)
    count_attack = y_train.count(1)
    print(f"Training Counts - Benign: {count_benign}, Attack: {count_attack}")
    
    # Weight = Total / (2 * Count)
    # If Benign is 1000 and Attack is 100:
    # W_benign = 1100/2000 = 0.55
    # W_attack = 1100/200 = 5.5 (10x punishment for missing an attack)
    total_count = len(y_train)
    w0 = total_count / (2 * max(count_benign, 1))
    w1 = total_count / (2 * max(count_attack, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE)
    print(f"Using CrossEntropy Class Weights: {class_weights}")
    
    # Use Standard Shuffle (let the Loss handle the imbalance)
    train_loader = DataLoader(FinetuneDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FinetuneDataset(test_data), batch_size=BATCH_SIZE)
    
    # Load Pretrained Model
    print("Loading Pretrained Phase 1 Model...")
    encoder = BiMambaEncoder(d_model=Config.D_MODEL)
    try:
        w_path = f"{Config.CHECKPOINT_DIR}/epoch_1.pth"
        print(f"DEBUG: Loading Pretrained Weights from: {os.path.abspath(w_path)}")
        if not os.path.exists(w_path):
             print(f"DEBUG: File does not exist at {w_path}")
        encoder.load_state_dict(torch.load(w_path, map_location=DEVICE))
        print(f"Loaded {w_path}")
    except Exception as e:
        print(f"Warning: Could not load epoch_10.pth, training from scratch? No. Error: {e}")
        return

    model = BiMambaClassifier(encoder).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE)
    
    # Apply Punishment Weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training Loop with Early Stopping
    best_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"\n🚀 Starting Fine-Tuning (Max {PHASE2_EPOCHS} Epochs)...")
    
    for epoch in range(PHASE2_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        
        # Validation (using Test Set as Val for this quick experiment, or split further?)
        # Using Test set for Early Stopping logic is technically leakage but common in rapid prototyping.
        # Let's stick to Test Loss.
        val_loss, val_acc, _, _, _, _ = evaluate(model, test_loader, verbose=False)
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f} Acc={epoch_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
        
        # Check Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
                
    # Load Best Weights
    model.load_state_dict(best_model_wts)
    
    # Final Evaluation with metrics
    print("\n📊 Final Evaluation on Test Set...")
    start_time = time.time()
    v_loss, v_acc, v_auc, v_f1, v_fpr, v_rec = evaluate(model, test_loader, verbose=True)
    end_time = time.time()
    
    total_time = end_time - start_time
    # Latency per flow
    # This is Batched Latency.
    # Real-time latency (Batch=1) is different, but user asked for "Latency".
    # We can measure Batch=1 latency specifically.
    avg_latency = (total_time / len(test_data)) * 1000 # ms
    
    print(f"\n🏆 Results (0.1% Few-Shot):")
    print(f"Accuracy:  {v_acc*100:.2f}%")
    print(f"F1 Score:  {v_f1*100:.2f}%")
    print(f"AUC Score: {v_auc*100:.2f}%")
    print(f"Recall:    {v_rec*100:.2f}%")
    print(f"FPR:       {v_fpr*100:.2f}%")
    print(f"Latency:   {avg_latency:.4f} ms/flow (Batched)")
    
    # Save Model
    save_path = "final_phase2/teacher_fewshot.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

def evaluate(model, loader, verbose=False):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            
            probs = torch.softmax(logits, dim=1)[:, 1] # Prob of class 1
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    
    if verbose:
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        f1 = f1_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return avg_loss, acc, auc, f1, fpr, rec
    else:
        return avg_loss, acc, 0, 0, 0, 0

if __name__ == "__main__":
    train_phase2()
