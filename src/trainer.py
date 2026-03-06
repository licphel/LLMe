import shutil
import sys

import torch
from torch.utils.data import Dataset
import time
from pathlib import Path
import signal


class Trainer:
    def __init__(self, model, train_loader, lr=1e-3, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=1,
            threshold=0.05,
            cooldown=1,
            min_lr=1e-6,
        )

        self.step = 0
        self.epoch = 0
        self.losses = []
        self.stop_training = False
        self.patience = 10
        self.min_delta = 0.01
        self.best_loss = float("inf")
        self.wait = 0
        
        self.printer = TrainerPrinter()

        # ctrl+c early stop and save.
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        self.stop_training = True

    def check_early_stopping(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(
                    f"\n   Early stopping triggered! Loss didn't improve for {self.patience} epochs."
                )
                return True
        return False

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        recent_losses = []
        recent_grad_norms = []
        recent_mask_avgs = []
        recent_perplexities = []

        for batch_idx, batch in enumerate(self.train_loader):
            if self.stop_training:
                print("\n   Stopping training as requested...")
                return total_loss / (batch_idx + 1) if batch_idx > 0 else 0

            x, y, mask = batch
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)

            logits, per_token_loss = self.model(x, y)  # per_token_loss shape: [B, T]

            masked_loss = per_token_loss * mask.float()
            loss = masked_loss.sum() / mask.sum()

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.8)
            # calc grad norm
            total_grad_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm**0.5

            self.optimizer.step()

            total_loss += loss.item()
            self.losses.append(loss.item())

            # collect stats
            recent_losses.append(loss.item())
            recent_grad_norms.append(total_grad_norm)
            recent_mask_avgs.append(mask.float().mean().item())

            # perplexity = exp(loss)
            perplexity = torch.exp(loss).item()
            recent_perplexities.append(perplexity)

            if self.step % 25 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0

                window = min(100, len(recent_losses))
                avg_loss = sum(recent_losses[-window:]) / window
                avg_grad = sum(recent_grad_norms[-window:]) / window
                avg_mask = sum(recent_mask_avgs[-window:]) / window
                avg_ppl = sum(recent_perplexities[-window:]) / window

                current_lr = self.optimizer.param_groups[0]["lr"]

                total_param_norm = 0
                for p in self.model.parameters():
                    param_norm = p.data.norm(2)
                    total_param_norm += param_norm.item() ** 2
                total_param_norm = total_param_norm**0.5

                grad_param_ratio = avg_grad / (total_param_norm + 1e-8)

                status_dict = {
                    'epoch': self.epoch + 1,
                    'step': self.step,
                    'loss': loss.item(),
                    'avg_loss': avg_loss,
                    'ppl': torch.exp(loss).item(),
                    'mask': mask.float().mean().item(),
                    'lr': current_lr,
                    'grad': total_grad_norm,
                    'gp_ratio': grad_param_ratio,
                    'speed': steps_per_sec,
                    'progress': batch_idx,
                    'total': len(self.train_loader)
                }
                
                self.printer.print_status(status_dict)

            self.step += 1

        self.epoch += 1
        print()

        avgloss = total_loss / len(self.train_loader)
        self.scheduler.step(avgloss)
        return avgloss

    def train(self, epochs, save_dir: Path, resume_from: int = 0):
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n Training on {self.device}")
        print(f"   Args: {self.model.count_parameters():,}")
        print(f"   Batch: {len(self.train_loader)}")
        print(f"   Steps: {len(self.train_loader) * (epochs - resume_from)}")
        print(f"   Resume from epoch {resume_from}")
        print("=" * 60)
        print("   Press Ctrl+C to stop training gracefully")
        print("=" * 60)

        for epoch in range(resume_from, epochs):
            if self.stop_training:
                print(f"\nTraining stopped by user at epoch {epoch}")
                self.save_model(f"{save_dir}/interrupted_epoch_{epoch}.pt")
                break

            print(f"\n Epoch {epoch+1}/{epochs}")

            avg_loss = self.train_epoch()
            print(f"\n   Avg loss: {avg_loss:.4f}")

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_model(f"{save_dir}/best.pt")
                print(f"   Best saved (loss={avg_loss:.4f})")

            if (epoch + 1) % 5 == 0:
                self.save_model(f"{save_dir}/epoch_{epoch+1}.pt")
                print(f"   Checkpoint saved at epoch {epoch+1}")

            if self.check_early_stopping(avg_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                self.save_model(f"{save_dir}/early_stop_epoch_{epoch+1}.pt")
                break

        if not self.stop_training:
            self.save_model(f"{save_dir}/final.pt")
            print(f"\n Training succeed.")
        else:
            print(f"\n Training interrupted.")

    def save_model(self, path):
        state_dict = self.model.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if not k.endswith(".causal_mask")
        }

        torch.save(
            {
                "model_state_dict": filtered_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "losses": self.losses,
                "best_loss": self.best_loss,
                "config": {
                    "vocab_size": self.model.vocab_size,
                    "dim": self.model.dim,
                    "n_layers": len(self.model.blocks),
                    "max_seq_len": self.model.max_seq_len,
                },
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.losses = checkpoint["losses"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        
        
class TrainerPrinter:
    def __init__(self):
        self.last_line_length = 0
        
    def print_status(self, d):
        terminal_width = shutil.get_terminal_size().columns
        
        status = (
            f"\r| S={d['step']:<6} "
            f"| L={d['loss']:<6.4f}({d['avg_loss']:<6.4f}) "
            f"| P={d['ppl']:<4.0f} "
            f"| M={d['mask']:<4.2f} "
            f"| LR={d['lr']:<.2e} "
            f"| G={d['grad']:<5.2f} "
            f"| G/P={d['gp_ratio']:<.3f} "
            f"| {d['speed']:<4.1f}sp/s "
            f"| {d['progress']:<4}/{d['total']} "
        )

        print(status)
       
        self.last_line_length = len(status)
