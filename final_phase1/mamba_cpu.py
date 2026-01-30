
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaBlockCPU(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        """
        x: (B, L, D) output of prev layer
        """
        batch, seq_len, d_model = x.shape
        
        # 1. Project to inner dim
        xz = self.in_proj(x) # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1) # (B, L, d_inner) each

        # 2. Conv1d
        # Rearrange to (B, d_inner, L) for Conv1d
        x_conv = x_inner.permute(0, 2, 1) 
        x_conv = self.conv1d(x_conv)[:, :, :seq_len] # Causal padding check
        x_conv = x_conv.permute(0, 2, 1) # Back to (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # 3. SSM (Selective Scan) logic in Python
        # Compute dt, B, C
        x_dbl = self.x_proj(x_conv) # (B, L, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt) # (B, L, d_inner)
        dt = F.softplus(dt)   # Delta > 0

        # Discretize A
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        # We need to run the recurrence: h_t = A_bar * h_{t-1} + B_bar * x_t
        # y_t = C * h_t + D * x_t
        
        # Prepare params for scan
        # This loop is SLOW in python but correct
        
        y_ssm = []
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device) # Hidden state
        
        for t in range(seq_len):
            # Per timestep
            dt_t = dt[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            dA = torch.exp(A * dt_t)         # (B, d_inner, d_state)
            
            x_t = x_conv[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            B_t = B_ssm[:, t, :].unsqueeze(1)   # (B, 1, d_state)
            
            # dB = dt * B
            dB = dt_t * B_t # (B, d_inner, d_state)
            
            # Update h: h = dA * h + dB * x
            h = dA * h + dB * x_t 
            
            # Compute y: y = C * h
            C_t = C_ssm[:, t, :].unsqueeze(-1) # (B, d_state, 1)
            # h is (B, d_inner, d_state). C is (B, d_state, 1). 
            # We want (B, d_inner). 
            # y_t = h @ C
            y_t = torch.matmul(h, C_t).squeeze(-1) # (B, d_inner)
            
            y_ssm.append(y_t)
            
        y_ssm = torch.stack(y_ssm, dim=1) # (B, L, d_inner)
        
        # Add D residual
        y_ssm = y_ssm + (x_conv * self.D)
        
        # 4. Gating
        out = y_ssm * F.silu(z)
        out = self.out_proj(out)
        
        return out

class BiMambaCPU(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        # Tokenizer matches model.py
        from model import PacketEmbedder
        self.tokenizer = PacketEmbedder(d_model)
        
        self.layers = nn.ModuleList([
            MambaBlockCPU(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        self.layers_rev = nn.ModuleList([
            MambaBlockCPU(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Heads
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128)
        )
        self.recon_head = nn.Linear(d_model, 5)

    def forward(self, x):
        x_emb = self.tokenizer(x)
        feat = x_emb
        
        for fwd, bwd in zip(self.layers, self.layers_rev):
            out_f = fwd(feat)
            
            feat_rev = torch.flip(feat, dims=[1])
            out_b = bwd(feat_rev)
            out_b = torch.flip(out_b, dims=[1])
            
            feat = self.norm(out_f + out_b + feat)
            
        global_rep = feat.mean(dim=1)
        z = self.proj_head(global_rep)
        return z, None # Not needed for eval
