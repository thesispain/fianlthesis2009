
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# Tokenizer (Same as Phase 1/3)
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

class LearnedEarlyExitStudent(nn.Module):
    """
    Learned Early Exit (2-Classifier) Strategy.
    Training: Explores positions 3-20.
    Inference: Checks ONLY 'Learned_Pos' and 'Final_Pos'.
    """
    def __init__(self, d_model=256, min_exit=3, max_exit=20, max_len=32):
        super().__init__()
        self.d_model = d_model
        self.min_exit = min_exit
        self.max_exit = max_exit
        
        self.tokenizer = PacketEmbedder(d_model)
        
        # Unidirectional Backbone (3 Layers for speed)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(3) 
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # Early Classifiers
        self.early_classifiers = nn.ModuleList([
            nn.Linear(d_model, 2) for _ in range(min_exit, max_exit + 1)
        ])
        
        # Final Classifier
        self.final_classifier = nn.Linear(d_model, 2)
        
        # Halting Network (Confidence Estimator)
        self.halting = nn.Linear(d_model, 1)
        
        # State
        self.register_buffer('learned_exit_pos', torch.tensor((min_exit + max_exit) // 2))
    
    def forward_train(self, x):
        """Returns logits for ALL positions to allow training."""
        x = self.tokenizer(x)
        features = x
        for layer in self.layers:
            features = layer(features)
        
        batch_size, seq_len, _ = features.shape
        
        # 1. Halting Scores
        halt_scores = torch.sigmoid(self.halting(features))
        
        # 2. Early Exits
        early_outputs = {}
        for i, pos in enumerate(range(self.min_exit, self.max_exit + 1)):
            if pos < seq_len:
                # Features at 'pos'
                feat_at_pos = self.norm(features[:, pos, :]) 
                early_outputs[pos] = self.early_classifiers[i](feat_at_pos)
        
        # 3. Final Exit
        final_feat = self.norm(features[:, -1, :])
        final_logits = self.final_classifier(final_feat)
        
        return final_logits, halt_scores, early_outputs

    def forward(self, x, inference=False):
        if not inference:
            return self.forward_train(x)
        
        # Efficient Inference (2 Checks)
        x = self.tokenizer(x)
        features = x
        for layer in self.layers:
            features = layer(features)
            
        # Check Learned Position
        pos = self.learned_exit_pos.item()
        feat_early = self.norm(features[:, pos, :])
        confidence = torch.sigmoid(self.halting(feat_early))
        
        # Threshold Logic (e.g. 0.9) - usually handled outside or hardcoded
        # Here we return everything needed for the decision loop
        logits_early = self.early_classifiers[pos - self.min_exit](feat_early)
        
        feat_final = self.norm(features[:, -1, :])
        logits_final = self.final_classifier(feat_final)
        
        return logits_early, logits_final, confidence, pos
