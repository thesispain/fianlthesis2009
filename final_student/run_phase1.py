
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure Root is in path
sys.path.append(os.getcwd())

from final_student.config import Config
from final_phase1.dataset import PretrainDataset # Reuse dataset
from final_student.model import UniMambaEncoder
from final_phase1.trainer import train_one_epoch, save_checkpoint # Reuse trainer logic
from final_phase1.telegram.telegram_notifier import catch_all_outputs

def main():
    catch_all_outputs()
    print(f"🚀 Starting Phase 1: Self-Supervised Pretraining")
    Config.print_config()
    
    # 1. Dataset
    dataset = PretrainDataset(Config.DATA_PATH)
    if len(dataset) == 0:
        print("❌ Dataset empty. Run generation scripts first.")
        return
        
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 2. Model
    print(f"🏗️ Building Uni-Mamba Model...")
    model = UniMambaEncoder(d_model=Config.D_MODEL).to(Config.DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    
    # 3. Loop
    for ep in range(Config.EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, ep+1)
        print(f"✅ Epoch {ep+1} Complete. Avg Loss: {loss:.4f}")
        
        # Save every epoch or just end?
        save_checkpoint(model, f"epoch_{ep+1}.pth")
        save_checkpoint(model, "latest.pth")
        
    print("🎉 Phase 1 Complete.")

if __name__ == "__main__":
    main()
