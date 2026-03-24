import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generals import Attention, FeedForward, LayerNorm, BaseModel

class DenseModelCore(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = Attention(dim, n_heads)
        self.ff = FeedForward(dim, dim * 4)
        self.ln1 = LayerNorm(dim)
        self.ln2 = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # attention -> feedforward -> ...
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class DenseModel(BaseModel):
   def __init__(self, vocab_size, dim, layers, heads, seqlen, dropout):
        super().__init__(DenseModelCore, vocab_size, dim, layers, heads, seqlen, dropout)
