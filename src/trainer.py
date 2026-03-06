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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 10
        )

        self.step = 0
        self.epoch = 0
        self.losses = []
        self.stop_training = False
        self.patience = 3
        self.min_delta = 0.01
        self.best_loss = float("inf")
        self.wait = 0

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

        for batch_idx, batch in enumerate(self.train_loader):
            if self.stop_training:
                print("\n   Stopping training as requested...")
                return total_loss / (batch_idx + 1) if batch_idx > 0 else 0

            if len(batch) == 3:
                x, y, mask = batch
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
            else:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                mask = None

            # forward
            _, loss = self.model(x, y)

            # loss mask
            if mask is not None:
                loss = loss * mask.float()
                loss = loss.sum() / mask.sum()

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            self.losses.append(loss.item())

            if self.step % 10 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"\r   Step {self.step}: loss={loss.item():.4f}, lr={self.scheduler.get_last_lr()[0]:.2e}, {steps_per_sec:.1f} steps/s",
                    end="",
                )

            self.step += 1

        self.epoch += 1
        return total_loss / len(self.train_loader)

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
