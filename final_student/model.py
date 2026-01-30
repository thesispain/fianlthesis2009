
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba 

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

class UniMambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        
        # Unidirectional Mamba: Forward Only ("Student")
        self.layers = nn.ModuleList([
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
        for layer in self.layers:
            # Standard Pre-Norm Mamba Block (or Post-Norm, trusting official impl)
            # The Mamba block usually handles residual internally? 
            # No, Mamba(x) returns x_out, usually need to add residual.
            # Let's verify Mamba usage pattern. 
            # Usually: out = mamba(norm(x)) + x
            # In BiMamba we did: norm(fwd + bwd + x).
            # Here:
            out = layer(feat)
            feat = self.norm(out + feat)
            
        # Global Representation (Mean Pooling)
        # For Uni-Mamba, maybe Last Token is better?
        # But for contrastive learning, Mean Pooling is robust.
        # User wants "Same things we did with bidirectional", so likely Mean Pooling.
        global_rep = feat.mean(dim=1)
        
        # 1. Contrastive Output (Vector)
        z = self.proj_head(global_rep)
        
        # 2. Reconstruction Output (Sequence)
        recon = self.recon_head(feat)
        
        return z, recon
