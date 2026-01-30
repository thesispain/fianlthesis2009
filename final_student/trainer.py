
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import sys

# Import Config (Relative or Absolute? We assume running from root)
try:
    from final_phase1.config import Config
except ImportError:
    # If running directly inside folder
    from config import Config

def hybrid_loss(z_i, z_j, x_recon, x_orig):
    """
    z_i, z_j: Output vectors from Contrastive Head (B, 128)
    x_recon: Reconstructed sequence (B, 32, 5)
    x_orig: Original sequence (B, 32, 5)
    """
    # 1. NT-Xent (Contrastive)
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Cosine Similarity Matrix (B, B)
    logits = torch.matmul(z_i, z_j.T) / Config.TEMP
    labels = torch.arange(z_i.size(0)).to(z_i.device)
    
    loss_cont = F.cross_entropy(logits, labels)
    
    # 2. Reconstruction (MSE)
    loss_recon = F.mse_loss(x_recon, x_orig)
    
    # 3. Weighted Sum
    total_loss = (Config.ALPHA * loss_cont) + ((1 - Config.ALPHA) * loss_recon)
    
    return total_loss, loss_cont.item(), loss_recon.item()

def train_one_epoch(model, loader, optimizer, epoch_idx):
    model.train()
    total_loss = 0
    total_cont = 0
    total_recon = 0
    steps = 0
    
    # Throttle: Update screen only every 30 seconds to prevent lag
    loop = tqdm(loader, desc=f"Epoch {epoch_idx}", mininterval=30.0)
    
    for x, x_aug in loop:
        x = x.to(Config.DEVICE)
        x_aug = x_aug.to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        # Forward Pass
        z_i, recon_i = model(x)
        z_j, recon_j = model(x_aug) 
        
        loss, l_c, l_r = hybrid_loss(z_i, z_j, recon_i, x)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_cont += l_c
        total_recon += l_r
        steps += 1
        
        # Update progress bar stats every 50 steps
        if steps % 50 == 0:
            loop.set_postfix({'Loss': f"{loss.item():.3f}", 'Cont': f"{l_c:.2f}", 'Recon': f"{l_r:.3f}"})
        
    avg_loss = total_loss / steps
    return avg_loss

def save_checkpoint(model, filename="model.pth"):
    if not os.path.exists(Config.CHECKPOINT_DIR):
        os.makedirs(Config.CHECKPOINT_DIR)
    path = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")
