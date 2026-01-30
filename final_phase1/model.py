
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba 

# We use the official Mamba implementation or our block?
# The user liked 'modules/mamba/block.py'. I will stick to a clean implementation here using official Mamba if available, 
# or a simplified block for clarity. 
# Let's use the standard nn.Module structure.

class PacketEmbedder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        # Continuous: Len, IAT
        # Categorical: Proto, Flags, Dir
        self.emb_proto = nn.Embedding(256, 32)
        self.emb_flags = nn.Embedding(64, 32)
        self.emb_dir = nn.Embedding(2, 8)
        
        self.proj_len = nn.Linear(1, 32)
        self.proj_iat = nn.Linear(1, 32)
        
        # 32+32+8+32+32 = 136
        self.fusion = nn.Linear(136, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, 32, 5) -> Proto, Len, Flags, IAT, Dir
        proto = x[:,:,0].long().clamp(0, 255)
        length = x[:,:,1].unsqueeze(-1)
        flags = x[:,:,2].long().clamp(0, 63)
        iat = x[:,:,3].unsqueeze(-1)
        direction = x[:,:,4].long().clamp(0, 1)
        
        e_p = self.emb_proto(proto)
        e_f = self.emb_flags(flags)
        e_d = self.emb_dir(direction)
        e_l = self.proj_len(length)
        e_i = self.proj_iat(iat)
        
        cat = torch.cat([e_p, e_f, e_d, e_l, e_i], dim=-1)
        return self.norm(self.fusion(cat))

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        
        # Bidirectional Mamba: One forward, one backward per layer
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        # We need to flip for backward logic manually if using stock Mamba
        self.layers_rev = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Heads
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128) # Contrastive Dim
        )
        self.recon_head = nn.Linear(d_model, 5) # Reconstruct 5 raw features
        
    def forward(self, x):
        x_emb = self.tokenizer(x)
        
        feat = x_emb
        for fwd, bwd in zip(self.layers, self.layers_rev):
            # Forward pass
            out_f = fwd(feat)
            # Backward pass (flip -> process -> flip back)
            feat_rev = torch.flip(feat, dims=[1])
            out_b = bwd(feat_rev)
            out_b = torch.flip(out_b, dims=[1])
            
            # Fuse (Add or Concat? Adding keeps dim)
            feat = self.norm(out_f + out_b + feat) # Residual + Norm
            
        # Global Representation (Mean Pooling)
        global_rep = feat.mean(dim=1)
        
        # 1. Contrastive Output (Vector)
        z = self.proj_head(global_rep)
        
        # 2. Reconstruction Output (Sequence)
        recon = self.recon_head(feat)
        
        return z, recon
