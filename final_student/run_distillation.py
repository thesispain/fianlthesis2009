
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score
from torch.utils.data import DataLoader, Dataset
import copy

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from final_student.config import Config
from final_student.model import UniMambaEncoder
from final_phase1.model import BiMambaEncoder
from final_phase2.run_fewshot import BiMambaClassifier, FinetuneDataset, load_and_split_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- CONFIG ---
DISTILL_EPOCHS = 30
PATIENCE = 3
TEMP = 4.0        # Softening Temperature (Higher = Softer)
ALPHA = 0.5       # Weight for Distillation Loss (0.5 means equal weight to Soft/Hard)
BATCH_SIZE = 128
LR = 1e-3         # Slightly higher LR for scratch training

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    KD Loss = alpha * KLDiv(Soft_Student, Soft_Teacher) + (1-alpha) * CE(Hard_Student, Labels)
    """
    # Soft Targets
    soft_targets = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    
    # KL Divergence (batchmean is standard)
    distill_loss = nn.KLDivLoss(reduction='batchmean')(soft_targets, soft_teacher) * (T * T)
    
    # Hard Targets
    student_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * distill_loss + (1.0 - alpha) * student_loss

def run_distillation():
    print(f"--- STARTING KNOWLEDGE DISTILLATION (T={TEMP}, Alpha={ALPHA}) ---")
    
    # 1. Load Data
    # We use the SAME split as Phase 2
    train_data, test_data = load_and_split_data()
    train_loader = DataLoader(FinetuneDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FinetuneDataset(test_data), batch_size=BATCH_SIZE)
    
    # 2. Load Teacher (Frozen)
    print("Loading Teacher Model...")
    teacher_encoder = BiMambaEncoder(d_model=256) # Config.D_MODEL might be diff if referencing diff config
    teacher_model = BiMambaClassifier(teacher_encoder)
    
    # Load Fine-Tuned Weights
    w_path = "final_phase2/teacher_fewshot.pth"
    # Note: We are in Organized_Final/final_student/, so path is relative to execution root
    # Ideally run from Organized_Final root: python final_student/run_distillation.py
    if not os.path.exists(w_path):
        w_path = "../final_phase2/teacher_fewshot.pth" # Fallback
        
    try:
        teacher_model.load_state_dict(torch.load(w_path, map_location=DEVICE))
        print(f"✅ Loaded Teacher: {w_path}")
    except Exception as e:
        print(f"❌ Failed to load Teacher: {e}")
        return

    teacher_model.to(DEVICE)
    teacher_model.eval() # Teacher is always in Eval mode
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    # 3. Initialize Student (Random)
    print("Initializing Student Model (Random Weights)...")
    student_encoder = UniMambaEncoder(d_model=256)
    student_model = BiMambaClassifier(student_encoder).to(DEVICE)
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR)
    
    # 4. Training Loop
    best_loss = float('inf')
    patience_counter = 0
    best_student_wts = copy.deepcopy(student_model.state_dict())
    
    for epoch in range(DISTILL_EPOCHS):
        student_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Teacher Output (No Grad)
            with torch.no_grad():
                teacher_logits = teacher_model(x)
            
            # Student Output
            optimizer.zero_grad()
            student_logits = student_model(x)
            
            # Loss
            loss = distillation_loss(student_logits, teacher_logits, y, TEMP, ALPHA)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(student_logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        
        # Validation
        val_loss, val_acc, _, _, _, _ = evaluate(student_model, test_loader)
        
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f} Acc={epoch_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
        
        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_student_wts = copy.deepcopy(student_model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
                
    # 5. Final Eval
    student_model.load_state_dict(best_student_wts)
    print("\n📊 Final Evaluation of Distilled Student...")
    v_loss, v_acc, v_auc, v_f1, v_fpr, v_rec = evaluate(student_model, test_loader, verbose=True)
    
    print(f"\n🏆 Results (Distilled Student):")
    print(f"Accuracy:  {v_acc*100:.2f}%")
    print(f"F1 Score:  {v_f1*100:.2f}%")
    print(f"AUC Score: {v_auc*100:.2f}%")
    
    # Save
    torch.save(student_model.state_dict(), "final_student/distilled_student.pth")
    print("Saved to final_student/distilled_student.pth")

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
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    
    if verbose:
        try: auc = roc_auc_score(all_labels, all_probs)
        except: auc = 0.5
        f1 = f1_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return avg_loss, acc, auc, f1, fpr, rec
    else:
        return avg_loss, acc, 0, 0, 0, 0

if __name__ == "__main__":
    run_distillation()
