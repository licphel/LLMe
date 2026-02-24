import torch
from torch.utils.data import Dataset, DataLoader
import time
from pathlib import Path

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = tokenizer.encode(text)
        self.n_samples = max(0, len(self.data) // seq_len - 1)
        print(f"Dataset: {len(self.data)} tokens -> {self.n_samples} samples")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        
        x = torch.tensor(self.data[start:end], dtype=torch.long)
        y = torch.tensor(self.data[start+1:end+1], dtype=torch.long)
        
        return x, y

class Trainer:
    def __init__(self, model, train_loader, lr=1e-3, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 10
        )
        
        self.step = 0
        self.epoch = 0
        self.losses = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # forward
            _, loss = self.model(x, y)
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.losses.append(loss.item())
            
            if self.step % 50 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                print(f"\r   Step {self.step}: loss={loss.item():.4f}, lr={self.scheduler.get_last_lr()[0]:.2e}, {steps_per_sec:.1f} steps/s", end='')
            
            self.step += 1
        
        self.epoch += 1
        return total_loss / len(self.train_loader)
    
    def train(self, epochs, save_dir='models'):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n Training on {self.device}")
        print(f"   Args: {self.model.count_parameters():,}")
        print(f"   Batch: {len(self.train_loader)}")
        print(f"   Steps: {len(self.train_loader) * epochs}")
        print("="*60)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n Epoch {epoch+1}/{epochs}")
            
            avg_loss = self.train_epoch()
            print(f"\n   Avg loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(f"{save_dir}/best.pt")
                print(f"   Best saved.")
            
            if (epoch + 1) % 5 == 0:
                self.save_model(f"{save_dir}/epoch_{epoch+1}.pt")
        
        self.save_model(f"{save_dir}/final.pt")
        print(f"\n Training succeed.")
    
    def save_model(self, path):
        state_dict = self.model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                            if not k.endswith('.causal_mask')}
        
        torch.save({
            'model_state_dict': filtered_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'losses': self.losses,
            'config': {
                'vocab_size': self.model.vocab_size,
                'dim': self.model.dim,
                'n_layers': len(self.model.blocks),
                'max_seq_len': self.model.max_seq_len
            }
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.losses = checkpoint['losses']