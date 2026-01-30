
import torch
import os

class Config:
    # Paths
    DATA_PATH = "../data/unswnb15_full/pretrain_50pct_benign.pkl"
    CHECKPOINT_DIR = "final_phase1/weights"  # User Request: Explicit 'weights' folder
    LOG_FILE = "final_phase1/training.log"
    
    # Model
    D_MODEL = 256
    INPUT_DIM = 5
    
    # Training
    BATCH_SIZE = 128  # Full GPU Power
    EPOCHS = 1       # Fast Retrain for Teacher Repair
    LR = 5e-4
    
    # Loss
    TEMP = 0.5   # NT-Xent Temperature
    ALPHA = 0.5  # Weight for Contrastive (vs Reconstruction)
    
    # Augmentation
    AUGMENT_RATIO = 0.45 # Increased to 45% (Harder task)
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def print_config():
        print(f"--- CONFIGURATION ---")
        print(f"Data: {Config.DATA_PATH}")
        print(f"Device: {Config.DEVICE}")
        print(f"Batch: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}")
        print(f"Alpha: {Config.ALPHA} (0.5 = Balanced Hybrid Loss)")
