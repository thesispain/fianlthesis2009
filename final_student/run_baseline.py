
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Add root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from final_student.models.baseline import BaselineStudent
from final_phase2.run_fewshot import FinetuneDataset, load_and_split_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128

def run():
    print("--- 🟢 RUNNING BASELINE (NO EXIT) ---")
    
    # 1. Data
    _, test_data = load_and_split_data() # Only need test
    test_loader = DataLoader(FinetuneDataset(test_data), batch_size=BATCH_SIZE)
    
    # 2. Model
    model = BaselineStudent(d_model=256).to(DEVICE)
    
    # 3. Load Weights (Transfer from Phase 4)
    # Phase 4 'distilled_student.pth' has 'encoder.tokenizer...'
    # Baseline has 'tokenizer...'
    ckpt = torch.load("final_student/distilled_student.pth", map_location=DEVICE)
    new_state = {}
    for k, v in ckpt.items():
        if k.startswith('encoder.proj_head'):
             new_state[k.replace('encoder.', '')] = v
        elif k.startswith('encoder.'):
            new_state[k.replace('encoder.', '')] = v
        elif k.startswith('head.'): 
            new_state[k] = v
    
    try:
        model.load_state_dict(new_state, strict=False)
        print("✅ Loaded Distilled Weights")
    except Exception as e:
        print(f"⚠️ Weight loading warning: {e}")
        
    # 4. Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    import time
    start = time.time()
    
    import numpy as np
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            # Probability of positive class (Attack) for AUC
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    end = time.time()
    latency = (end - start) / len(test_data) * 1000
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    print(f"🏆 Baseline Results: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Latency={latency:.4f} ms")
    
    return acc, f1, auc, latency

if __name__ == "__main__":
    run()
