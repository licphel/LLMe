import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta


class Attention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, "dim must be divisible by heads"

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.register_buffer("causal_mask", None)

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.causal_mask is None or self.causal_mask.shape[-1] != T:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            self.register_buffer("causal_mask", mask)

        attn = attn.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        attn = F.softmax(attn, dim=-1)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        return y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = Attention(dim, n_heads)
        self.ff = FeedForward(dim)
        self.ln1 = LayerNorm(dim)
        self.ln2 = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class LanguageModel(nn.Module):
    def __init__(
        self, vocab_size, dim=256, n_layers=4, n_heads=4, max_seq_len=512, dropout=0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.config = {
            "vocab_size": vocab_size,
            "dim": dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
        }

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        self.apply(self._init_weights)

        print(
            f"Model created: vocab_size={vocab_size}, dim={dim}, layers={n_layers}, heads={n_heads}"
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape

        assert T <= self.max_seq_len, f"{T} exceeds {self.max_seq_len}"

        token_emb = self.token_embedding(x)
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = self.embed_dropout(token_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), 
                targets.view(-1), 
                ignore_index=-1,
                reduction="none"
            )
            loss = loss.view(B, T)
            return logits, loss

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, top_p=0.95, 
             repetition_penalty=1.2, eos_token_id=None):
        self.eval()
        
        generated = []
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[0, -1, :] / temperature
            
            # rep penalty
            if repetition_penalty > 1.0 and generated:
                for token_id in set(generated[-10:]):
                    logits[token_id] /= repetition_penalty
            
            # top-k 
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float("-inf")
            
            # top-p 
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float("-inf")
            
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            generated.append(idx_next.item())
            
            idx = torch.cat([idx, idx_next.unsqueeze(0)], dim=1)
            
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break
        
        return idx

    @torch.no_grad()
    def generate_text(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40):
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        output_tensor = self.generate(input_tensor, max_new_tokens, temperature, top_k)
        return output_tensor[0].tolist()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
