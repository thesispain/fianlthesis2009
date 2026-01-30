
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from final_student.models.dynamic_exit import DynamicEarlyExitStudent
from final_phase2.run_fewshot import FinetuneDataset, load_and_split_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3

def run():
    print("--- 🔵 RUNNING DYNAMIC ACT (TRAINING + EVAL) ---")
    
    train_data, test_data = load_and_split_data()
    train_loader = DataLoader(FinetuneDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FinetuneDataset(test_data), batch_size=BATCH_SIZE)
    
    model = DynamicEarlyExitStudent(d_model=256).to(DEVICE)
    
    # Load Backbone
    ckpt = torch.load("final_student/distilled_student.pth", map_location=DEVICE)
    new_state = {}
    for k, v in ckpt.items():
        if k.startswith('encoder.'):
            # 'encoder.tokenizer.abc' -> 'tokenizer.abc'
            new_state[k.replace('encoder.', '')] = v
            
    try:
        model.load_state_dict(new_state, strict=False)
        print("✅ Loaded Backbone Weights")
    except: pass
    
    # Train Halting
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    crit_cls = nn.CrossEntropyLoss()
    
    print("Training Halting Network...")
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward returns stack of logits
            logits_seq, halt_seq = model(x) # (B, L, 2), (B, L)
            
            # Loss: Classification at every step + Halting Sparsity?
            # Simple ACT: Maximize correctness at every step weighted by halting?
            # Or just train all classifiers?
            # Let's train classification at every step heavily
            
            B, L, C = logits_seq.shape
            # Flatten for loss
            loss_cls = crit_cls(logits_seq.view(-1, C), y.repeat_interleave(L))
            
            # Halting Loss (ACT Ponder Cost)
            # Encourages exiting early? N/A for simple thresholding training
            # Usually we train 'halting' via REINFORCE or simple supervised if we have oracle.
            # Simplified: Just train classifiers here.
            
            # Wait, Dynamic ACT usually uses a fixed geometric distribution or learnt.
            # Let's just train the classifiers to be good at every step.
            # And Halting to predict confidence (correctness).
            # Target for halting[t] = 1 if pred[t] == y, else 0
            
            preds_seq = torch.argmax(logits_seq, dim=2) # (B, L)
            target_halt = (preds_seq == y.unsqueeze(1)).float()
            loss_halt = nn.MSELoss()(halt_seq, target_halt)
            
            loss = loss_cls + loss_halt
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} Complete")
        
    # Evaluate (Inference Mode with Threshold)
    model.eval()
    all_preds = []
    all_labels = []
    exit_steps = []
    
    import time
    start = time.time()
    
    THRESHOLD = 0.9
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits_seq, halt_seq = model(x)
            
            batch_preds = []
            
            for b in range(x.size(0)):
                # Per sample logic simulating exit
                exited = False
                for t in range(logits_seq.size(1)):
                    conf = halt_seq[b, t].item()
                    if conf > THRESHOLD:
                        batch_preds.append(torch.argmax(logits_seq[b, t]).item())
                        exit_steps.append(t + 1)
                        exited = True
                        break
                if not exited:
                    batch_preds.append(torch.argmax(logits_seq[b, -1]).item())
                    exit_steps.append(32)
            
            all_preds.extend(batch_preds)
            all_labels.extend(y.cpu().numpy())
            
    end = time.time()
    # Note: Loop above is slow Python, so latency metric will be skewed.
    # We report 'simulated' latency based on avg steps.
    # 1 Step ~ 0.005 ms (Standard Mamba step)
    
    avg_step = np.mean(exit_steps)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"🏆 Dynamic Results: Acc={acc:.4f}, F1={f1:.4f}, Avg Step={avg_step:.2f}")
    return acc, f1, avg_step

if __name__ == "__main__":
    run()
