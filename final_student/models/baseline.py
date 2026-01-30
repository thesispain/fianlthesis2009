
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class PacketEmbedder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.start_emb = nn.Embedding(256, 32) # Just adapting, actually use same code
        self.emb_proto = nn.Embedding(256, 32)
        self.emb_flags = nn.Embedding(64, 32)
        self.emb_dir = nn.Embedding(2, 8)
        self.proj_len = nn.Linear(1, 32)
        self.proj_iat = nn.Linear(1, 32)
        self.fusion = nn.Linear(136, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        proto = x[:,:,0].long().clamp(0, 255)
        length = x[:,:,1].unsqueeze(-1)
        flags = x[:,:,2].long().clamp(0, 63)
        iat = x[:,:,3].unsqueeze(-1)
        direction = x[:,:,4].long().clamp(0, 1)
        cat = torch.cat([self.emb_proto(proto), self.emb_flags(flags), self.emb_dir(direction), self.proj_len(length), self.proj_iat(iat)], dim=-1)
        return self.norm(self.fusion(cat))

class BaselineStudent(nn.Module):
    """
    Standard Uni-Mamba (No Early Exit).
    Processes all 32 packets, then classifies.
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = PacketEmbedder(d_model)
        
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(3)
        ])
        
        # Projection Head (Matches Phase 4 Teacher/Student)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
        # Final Classifier (Matches Phase 4)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Add dropout if it was in config, Phase 4 had it
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        x = self.tokenizer(x) # (B, 32, D)
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        feat = x.mean(dim=1) 
        
        # Project
        z = self.proj_head(feat)
        
        return self.head(z)
