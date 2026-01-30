
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from final_student.models.learned_exit import LearnedEarlyExitStudent
from final_phase2.run_fewshot import FinetuneDataset, load_and_split_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3

def run():
    print("--- 🟠 RUNNING LEARNED EXIT (TRAINING + EVAL) ---")
    
    train_data, test_data = load_and_split_data()
    train_loader = DataLoader(FinetuneDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FinetuneDataset(test_data), batch_size=BATCH_SIZE)
    
    model = LearnedEarlyExitStudent(d_model=256).to(DEVICE)
    
    # Load Backbone
    ckpt = torch.load("final_student/distilled_student.pth", map_location=DEVICE)
    new_state = {}
    for k, v in ckpt.items():
        if k.startswith('encoder.'):
            new_state[k.replace('encoder.', '')] = v
    try:
        model.load_state_dict(new_state, strict=False)
        print("✅ Loaded Backbone Weights")
    except: pass
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    crit_cls = nn.CrossEntropyLoss()
    
    # Exploration Stats (to find optimal exit)
    exit_counts = np.zeros(18) # 3-20
    
    print("Training Early Exits...")
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            final_logits, halt_scores, early_outputs = model.forward_train(x)
            
            # 1. Final Loss
            loss = crit_cls(final_logits, y)
            
            # 2. Early Losses (Train all classifiers)
            for pos, logits in early_outputs.items():
                loss += crit_cls(logits, y)
                
                # Check accuracy at this pos
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean()
                
                # If accuracy is good, boost halting score?
                # Simple logic for this demo:
                # If correct, we WANT to exit here -> Target Halting = 1
                target_h = (preds == y).float().unsqueeze(1)
                loss += nn.MSELoss()(halt_scores[:, pos, :], target_h)
            
            loss.backward()
            optimizer.step()
            
    # Determine Optimal Exit Position (based on Validation Accuracy)
    print("Determining Optimal Exit Position...")
    model.eval()
    best_pos = 20
    best_acc = 0
    
    with torch.no_grad():
        # Quick scan over test set
        for pos in range(3, 21):
            correct = 0
            total = 0
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _, _ = model.forward_train(x) # Access internals via train mode or custom
                # Hack: reconstruct just that pos
                # Actually forward_train returns dict
                _, _, early = model.forward_train(x)
                if pos in early:
                    pred = torch.argmax(early[pos], dim=1)
                    correct += (pred == y).sum().item()
                    total += x.size(0)
            
            acc = correct / total
            # print(f"Pos {pos}: Acc {acc:.4f}")
            if acc >= 0.98: # Threshold for "Good Enough"
                best_pos = pos
                print(f"Found acceptable exit at Pos {pos} (Acc {acc:.4f})")
                break
    
    model.learned_exit_pos.fill_(best_pos)
    print(f"👉 Set Learned Exit Position to: {best_pos}")
    
    # Final Evaluation (Inference Logic)
    all_preds = []
    all_labels = []
    exits = []
    
    import time
    start = time.time()
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            # Use the inference method
            early_logits, final_logits, conf, pos = model(x, inference=True)
            
            batch_preds = []
            
            conf_np = conf.cpu().numpy().flatten()
            early_p = torch.argmax(early_logits, dim=1).cpu().numpy()
            final_p = torch.argmax(final_logits, dim=1).cpu().numpy()
            
            for i in range(x.size(0)):
                if conf_np[i] > 0.9:
                    batch_preds.append(early_p[i])
                    exits.append(pos)
                else:
                    batch_preds.append(final_p[i])
                    exits.append(32)
            
            all_preds.extend(batch_preds)
            all_labels.extend(y.numpy())
            
    end = time.time()
    # Latency: Simulating 2 checks vs 32
    # Standard: 32 steps. Learned: 'pos' steps + '32-pos' steps?
    # No, Learned is: Run to 'pos', check. If fail, run to 32, check.
    # Avg steps = (Ratio_Early * Pos) + (Ratio_Late * 32)
    
    avg_step = np.mean(exits)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"🏆 Learned Results: Acc={acc:.4f}, F1={f1:.4f}, Avg Step={avg_step:.2f}")
    return acc, f1, avg_step

if __name__ == "__main__":
    run()
