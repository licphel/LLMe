import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .generals import Attention, FeedForward, LayerNorm, BaseModel

EXPERTS = 16
TOPK = 2


class Router(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.prj = nn.Linear(dim, num_experts)
        self.top_k = top_k
        
    def forward(self, x):
        logits = self.prj(x)
        weights = torch.softmax(logits, dim=-1)
        
        top_k_values, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        
        # 归一化top-k
        top_k_values = top_k_values / top_k_values.sum(dim=-1, keepdim=True)
        
        return top_k_values, top_k_indices
      
      
class MoELayer(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList([FeedForward(dim, dim) for _ in range(num_experts)])
        self.router = Router(dim, num_experts, top_k)
        self.num_experts = num_experts
        self.top_k = top_k
        
    def forward(self, x):
        B, S, D = x.shape
        N = B * S  # token cnt
        
        # weights: [B, S, top_k], indices: [B, S, top_k]
        weights, indices = self.router(x)
        
        x_flat = x.view(N, D)  # [N, D]
        indices_flat = indices.view(N, self.top_k)  # [N, top_k]
        weights_flat = weights.view(N, self.top_k)  # [N, top_k]
        
        outputs = torch.zeros(N, D, device=x.device)
        
        for expert_id in range(self.num_experts):
            mask = (indices_flat == expert_id)
            
            if not mask.any():
                continue
            
            token_mask = mask.any(dim=-1)  # [N]
            
            token_idx = torch.where(token_mask)[0]  # [M]

            
            flat_mask = mask.view(-1)  # [N*top_k]
            
            token_idx_expanded = torch.arange(N, device=x.device).repeat_interleave(self.top_k)  # [N*top_k]
            
            selected_token_idx = token_idx_expanded[flat_mask]  # [K]
            selected_weights = weights_flat.view(-1)[flat_mask]  # [K]
            
            unique_tokens, inverse = torch.unique(selected_token_idx, return_inverse=True)
            
            combined_weights = torch.zeros(len(unique_tokens), device=x.device)
            combined_weights.scatter_add_(0, inverse, selected_weights)
            
            selected_x = x_flat[unique_tokens]  # [U, D]
            
            expert_out = self.experts[expert_id](selected_x)  # [U, D]
            
            # 加权
            weighted_out = expert_out * combined_weights.unsqueeze(-1)  # [U, D]
            
            outputs[unique_tokens] += weighted_out
        
        return outputs.view(B, S, D)
      

class MoECore(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.attn = Attention(dim, n_heads)
        self.moe = MoELayer(dim, EXPERTS, TOPK)
        self.ln1 = LayerNorm(dim)
        self.ln2 = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        
        moe_input = self.ln2(x)
        moe_output = self.moe(moe_input)
        
        x = x + self.dropout(moe_output)
        
        return x


class MoEModel(BaseModel):
   def __init__(self, vocab_size, dim, layers, heads, seqlen, dropout):
        super().__init__(MoECore, vocab_size, dim, layers, heads, seqlen, dropout)
