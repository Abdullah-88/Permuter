import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from sinkhorn_transformer import SinkhornTransformer



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)






class PermuterGatingUnit(nn.Module):
    def __init__(self,d_model,d_ffn,dropout):
        super().__init__()
        self.proj = nn.Linear(d_model,d_model)     
        self.Perm = SinkhornTransformer(
			    dim = d_model,
			    heads = 8,
			    depth = 1,
			    bucket_size = 64
			)

	
       

    def forward(self, x):
        u, v = x, x 
        u = self.proj(u)  
        v = self.Perm(v)
        out = u * v
        return out


class PermuterBlock(nn.Module):
    def __init__(self, d_model, d_ffn,dropout):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.pgu = PermuterGatingUnit(d_model,d_ffn,dropout)
        self.ffn = FeedForward(d_model,d_ffn,dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.pgu(x)   
        x = x + residual      
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        out = x + residual
        return out



class Permuter(nn.Module):
    def __init__(self, d_model, d_ffn, num_layers,dropout):
        super().__init__()
        
        self.model = nn.Sequential(
            *[PermuterBlock(d_model,d_ffn,dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








