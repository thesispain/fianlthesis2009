
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class PacketEmbedder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
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

class DynamicEarlyExitStudent(nn.Module):
    """
    Dynamic Early Exit (ACT) Strategy.
    Check halting at EVERY step. Exit when confidence > threshold.
    """
    def __init__(self, d_model=256, max_len=32):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        self.tokenizer = PacketEmbedder(d_model)
        
        # Unidirectional Backbone
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(3)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # Shared Classifier (Feature space is consistent)
        # OR Separate Classifiers per step? 
        # Standard ACT usually shares weight if dimensions match.
        self.classifier = nn.Linear(d_model, 2)
        
        # Halting Network
        self.halting = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.tokenizer(x)
        features = x
        for layer in self.layers:
            features = layer(features)
        
        # features: (B, L, D)
        B, L, D = features.shape
        
        all_logits = []
        all_halt_scores = []
        
        for t in range(L):
            ft = self.norm(features[:, t, :])
            halt_score = torch.sigmoid(self.halting(ft))
            logits = self.classifier(ft)
            
            all_halt_scores.append(halt_score)
            all_logits.append(logits)
            
        # Return as tensors
        return torch.stack(all_logits, dim=1), torch.cat(all_halt_scores, dim=1)
