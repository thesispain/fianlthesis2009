
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import sys
import os
import time
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 0. CONFIG & SETUP
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

DATA_PATH = "data/pretrain_50pct_benign.pkl"  # 50% Benign Subset
MAX_EPOCHS = 5
BATCH_SIZE = 128
D_MODEL = 256
MAX_LEN = 32
LR = 5e-4
SAVE_PATH = "weights/mamba_scratch_best.pth"
if not os.path.exists("weights"): os.makedirs("weights")

# ==========================================
# 1. MODULE DEFINITIONS (Self Contained)
# ==========================================

# --- CutMix Augmenter ---
class CutMixAugmenter(nn.Module):
    def __init__(self, segment_len=10):
        super().__init__()
        self.segment_len = segment_len

    def forward(self, x):
        B, S, F = x.shape
        indices = torch.randperm(B, device=x.device)
        x_B = x[indices]
        valid_starts = S - self.segment_len
        if valid_starts <= 0: return x
        starts = torch.randint(0, valid_starts, (B,), device=x.device)
        x_aug = x.clone()
        for i in range(B):
            st = starts[i]
            en = st + self.segment_len
            x_aug[i, st:en, :] = x_B[i, st:en, :]
        return x_aug

# --- Packet Embedder ---
class PacketEmbedder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.embed_proto = nn.Embedding(num_embeddings=256, embedding_dim=16)
        self.embed_len = nn.Linear(in_features=1, out_features=32)
        self.embed_flags = nn.Embedding(num_embeddings=64, embedding_dim=16)
        self.embed_iat = nn.Linear(in_features=1, out_features=32)
        self.embed_dir = nn.Embedding(num_embeddings=2, embedding_dim=8)
        fusion_dim = 16 + 32 + 16 + 32 + 8
        self.fusion_layer = nn.Linear(fusion_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        proto = x[:, :, 0].long()
        log_len = x[:, :, 1].unsqueeze(-1)
        flags = x[:, :, 2].long()
        log_iat = x[:, :, 3].unsqueeze(-1)
        direction = x[:, :, 4].long()
        
        e_proto = self.embed_proto(proto)
        e_len = self.embed_len(log_len)
        e_flags = self.embed_flags(flags)
        e_iat = self.embed_iat(log_iat)
        e_dir = self.embed_dir(direction)
        
        concatenated = torch.cat((e_proto, e_len, e_flags, e_iat, e_dir), dim=-1)
        fused = self.fusion_layer(concatenated)
        output = self.layer_norm(fused)
        return output

# --- Mamba Block ---
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    print("Error: mamba_ssm not installed. Please install it.")
    sys.exit(1)

class MambaBlock(nn.Module):
    def __init__(self, d_model=256, d_state=16, expand=2, d_conv=4, dropout=0.1, bidirectional=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.bidirectional = bidirectional
        
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = self.d_model // 16
        
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, 
            kernel_size=d_conv, groups=self.d_inner, 
            padding=d_conv - 1, bias=True
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bidirectional: x = torch.flip(x, [1])
        residual = x
        x = self.norm(x)
        x_proj = self.in_proj(x)
        x_ssm, x_gate = x_proj.chunk(2, dim=-1)
        
        x_ssm_conv = x_ssm.transpose(1, 2)
        x_ssm_conv = self.conv1d(x_ssm_conv)[:, :, :x.shape[1]]
        x_ssm_conv = x_ssm_conv.transpose(1, 2)
        x_ssm_act = self.act(x_ssm_conv)
        
        x_dbl = self.x_proj(x_ssm_act)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        
        u = x_ssm_act.transpose(1, 2)
        delta = dt.transpose(1, 2)
        A = -torch.exp(self.A_log)
        B_t = B_ssm.transpose(1, 2)
        C_t = C_ssm.transpose(1, 2)
        
        y_ssm = selective_scan_fn(u, delta, A, B_t, C_t, self.D, z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        y_ssm = y_ssm.transpose(1, 2)
        
        x_gate_act = self.act(x_gate)
        x_fused = y_ssm * x_gate_act
        out = self.dropout(self.out_proj(x_fused))
        final_output = out + residual
        if self.bidirectional: final_output = torch.flip(final_output, [1])
        return final_output

# --- Hybrid Mamba Teacher ---
class HybridMambaTeacher(nn.Module):
    def __init__(self, d_model=256, input_dim=5):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model=d_model)
        self.backbone = nn.Sequential(
            MambaBlock(d_model=d_model, bidirectional=True, d_state=16, expand=2),
            MambaBlock(d_model=d_model, bidirectional=True, d_state=16, expand=2),
            MambaBlock(d_model=d_model, bidirectional=True, d_state=16, expand=2),
            MambaBlock(d_model=d_model, bidirectional=True, d_state=16, expand=2)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128)
        )
        self.reconstruction_head = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x_emb = self.tokenizer(x)
        features = self.backbone(x_emb)
        z = self.projection_head(features[:, -1, :])
        x_recon = self.reconstruction_head(features)
        return z, x_recon, features

# --- Loss Function ---
def hybrid_loss(z_i, z_j, x_recon, x_orig, temp=0.5, alpha=0.5):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    sim = torch.mm(z_i, z_j.t()) / temp
    labels = torch.arange(z_i.size(0)).to(z_i.device)
    loss_cont = F.cross_entropy(sim, labels)
    loss_recon = F.mse_loss(x_recon, x_orig)
    return (alpha * loss_cont) + ((1 - alpha) * loss_recon)

# ==========================================
# 2. DATASET & EXECUTION
# ==========================================

class ContrastiveDataset(Dataset):
    def __init__(self, data, max_len=32):
        self.flows = []
        for row in data:
            if 'features' in row: f = row['features']
            else: f = np.stack([row['proto'], row['len'], row['flags'], row['iat'], row['direction']], axis=1)
            L = f.shape[0]
            if L > max_len: f = f[:max_len]
            elif L < max_len: f = np.vstack([f, np.zeros((max_len - L, 5))])
            self.flows.append(torch.from_numpy(f).float())
    def __len__(self): return len(self.flows)
    def __getitem__(self, idx): return self.flows[idx]

print(f"Loading {DATA_PATH}...")
all_data = []
try:
    with open(DATA_PATH, 'rb') as f:
        while True:
            try: all_data.extend(pickle.load(f))
            except EOFError: break
    print(f"Loaded {len(all_data)} flows. (50% Benign Subset)")
except FileNotFoundError:
    print(f"ERROR: File {DATA_PATH} not found.")
    sys.exit(1)

# Split for Validation (Monitoring Memorization)
train_data = all_data[:int(0.9 * len(all_data))]
val_data = all_data[int(0.9 * len(all_data)):]
print(f"Train Set: {len(train_data)} | Val Set: {len(val_data)}")

train_dl = DataLoader(ContrastiveDataset(train_data, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dl = DataLoader(ContrastiveDataset(val_data, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)

print(f"Batches per Epoch: {len(train_dl)}")
print(f"Estimated Time per Epoch (at 0.2s/batch): ~{len(train_dl)*0.2/60:.1f} minutes")

model = HybridMambaTeacher(d_model=D_MODEL).to(DEVICE)
augmenter = CutMixAugmenter().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"\nStarting Pre-training for {MAX_EPOCHS} epochs...")
best_val_loss = float('inf')
patience = 0
MAX_PATIENCE = 1

for ep in range(MAX_EPOCHS):
    model.train()
    train_loss = 0
    start_time = time.time()
    for i, x in enumerate(train_dl):
        x = x.to(DEVICE)
        x_aug1 = augmenter(x); x_aug2 = augmenter(x)
        z1, rec1, _ = model(x_aug1)
        z2, rec2, _ = model(x_aug2)
        loss = hybrid_loss(z1, z2, rec1, x, alpha=0.5)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss += loss.item()
        if i % 100 == 0: print(f"  Ep {ep+1} Batch {i}/{len(train_dl)} Loss={loss.item():.4f}", end='\r')
            
    avg_train_loss = train_loss / len(train_dl)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x in val_dl:
            x = x.to(DEVICE)
            x_aug1 = augmenter(x); x_aug2 = augmenter(x)
            z1, rec1, _ = model(x_aug1)
            z2, rec2, _ = model(x_aug2)
            loss = hybrid_loss(z1, z2, rec1, x, alpha=0.5)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_dl)
    duration = time.time() - start_time
    print(f"\nEpoch {ep+1}/{MAX_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {duration/60:.1f}m")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Saved Best Model")
        patience = 0
    else:
        print(f"  ! Validation Loss Increased (Best: {best_val_loss:.4f})")
        patience += 1
        if patience >= MAX_PATIENCE:
            print("  STOPPING EARLY: Validation Loss isn't improving.")
            break

print("\nTraining Complete.")
