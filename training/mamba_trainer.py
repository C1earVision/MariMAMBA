import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config.training_config import MambaTrainingConfig

mamba_config = MambaTrainingConfig()


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


class MambaTrainer:

    def __init__(
        self,
        model,
        learning_rate: float = mamba_config.learning_rate,
        weight_decay: float = mamba_config.weight_decay,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        attr_loss_weight: float = 0.1
    ):
        self.model = model.to(device)
        self.device = device
        self.attr_loss_weight = attr_loss_weight

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Per-tile cross entropy (reduction='none' for masking)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # L1 loss for attribute prediction (more stable than MSE for counting)
        self.attr_criterion = nn.L1Loss(reduction='none')

        self.ema = EMA(self.model, decay=mamba_config.ema_decay)

        self.train_losses = []
        self.val_losses = []
        self.epoch_losses = []

        print(f"\n{'='*70}")
        print(f"Mamba Trainer Initialized")
        print(f"{'='*70}")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  EMA decay: {self.ema.decay}")
        print(f"  Attr loss weight: {self.attr_loss_weight}")
        print(f"{'='*70}\n")

    def _masked_column_ce(self, predicted, target, seq_lens):
        """
        Compute masked cross-entropy over column predictions.
        
        Args:
            predicted: [B, L, H, C] logits
            target: [B, L, H] tile indices
            seq_lens: [B] actual sequence lengths
            
        Returns:
            Scalar loss
        """
        B, L, H, C = predicted.shape

        # Reshape for cross entropy: [B*L*H, C] vs [B*L*H]
        per_tile = self.criterion(
            predicted.reshape(-1, C),
            target.reshape(-1)
        ).reshape(B, L, H)

        # Average over tiles in column: [B, L]
        per_column = per_tile.mean(dim=-1)

        # Mask padding columns
        mask = torch.zeros(B, L, device=predicted.device)
        for i in range(B):
            mask[i, :seq_lens[i]] = 1.0

        masked = per_column * mask
        loss = masked.sum() / (mask.sum() + 1e-8)
        return loss

    def _masked_attr_loss(self, predicted_attrs, target_attrs, seq_lens):
        """
        Compute masked L1 loss for attribute prediction.
        
        Args:
            predicted_attrs: [B, L, num_attributes]
            target_attrs: [B, L, num_attributes]
            seq_lens: [B]
        """
        B, L, K = predicted_attrs.shape
        
        # Scale attributes to keep loss in a reasonable range
        scale = 0.3
        
        # L1 loss: [B, L, K]
        l1_error = self.attr_criterion(predicted_attrs * scale, target_attrs * scale)
        
        # Mean over attributes: [B, L]
        per_step = l1_error.mean(dim=-1)
        
        # Masking
        mask = torch.zeros(B, L, device=predicted_attrs.device)
        for i in range(B):
            mask[i, :seq_lens[i]] = 1.0
            
        masked = per_step * mask
        loss = masked.sum() / (mask.sum() + 1e-8)
        return loss

    def train_step(self, batch_data):
        self.model.train()

        input_cols, cond_seq, target_cols, seq_lens = batch_data
        input_cols = input_cols.to(self.device).long()
        cond_seq = cond_seq.to(self.device).float()
        target_cols = target_cols.to(self.device).long()
        seq_lens = seq_lens.to(self.device).long()

        # CFG Masking
        if self.model.training:
            # Important: even if we drop attributes for CFG, 
            # we should still try to predict the TRUE attributes for auxiliary loss
            # so we keep a copy of the true cond_seq
            true_cond_seq = cond_seq.clone()
            drop_mask = torch.rand(input_cols.shape[0], device=self.device) < 0.15
            cond_seq[drop_mask] = -1.0

        predicted_logits, predicted_attrs = self.model(input_cols, cond_seq)
        
        # 1. Standard Cross Entropy Loss
        ce_loss = self._masked_column_ce(predicted_logits, target_cols, seq_lens)
        
        # 2. Auxiliary Attribute Loss
        attr_loss = self._masked_attr_loss(predicted_attrs, true_cond_seq, seq_lens)
        
        # Total Loss
        total_loss = ce_loss + self.attr_loss_weight * attr_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.ema.update(self.model)

        return {
            'total': total_loss.item(),
            'ce': ce_loss.item(),
            'attr': attr_loss.item()
        }

    def validate(self, val_loader: DataLoader, use_ema: bool = True) -> float:
        if use_ema:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                input_cols, cond_seq, target_cols, seq_lens = batch_data
                input_cols = input_cols.to(self.device).long()
                cond_seq = cond_seq.to(self.device).float()
                target_cols = target_cols.to(self.device).long()
                seq_lens = seq_lens.to(self.device).long()

                predicted_logits, predicted_attrs = self.model(input_cols, cond_seq)
                
                ce_loss = self._masked_column_ce(predicted_logits, target_cols, seq_lens)
                attr_loss = self._masked_attr_loss(predicted_attrs, cond_seq, seq_lens)
                
                loss = ce_loss + self.attr_loss_weight * attr_loss
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)

        if use_ema:
            self.ema.restore(self.model)

        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = mamba_config.num_epochs,
        save_interval: int = mamba_config.save_interval,
        save_path: str = mamba_config.save_path
    ):
        print("=" * 70)
        print("TRAINING MAMBA MODEL")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Epochs: {num_epochs}")

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=self.optimizer.param_groups[0]['lr'] * 0.01
        )

        best_val_loss = float('inf')
        patience = mamba_config.patience
        patience_counter = 0

        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_ce = 0
            epoch_attr = 0
            self.model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_data in progress_bar:
                losses = self.train_step(batch_data)
                epoch_loss += losses['total']
                progress_bar.set_postfix({
                    'ce': f"{losses['ce']:.4f}",
                    'attr': f"{losses['attr']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })

            avg_epoch_loss = epoch_loss / len(train_loader)
            self.epoch_losses.append(avg_epoch_loss)
            self.scheduler.step()

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(save_path.replace('.pth', '_best.pth'))
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_epoch_loss:.4f} | "
                      f"Val: {val_loss:.4f} | Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(f"\n⚠ Early stopping triggered at epoch {epoch+1}! "
                          f"No improvement for {patience} epochs.")
                    break
            else:
                print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_epoch_loss:.4f}")

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(save_path.replace('.pth', f'_epoch{epoch+1}.pth'))

        self.save_checkpoint(save_path)
        print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch_losses': self.epoch_losses,
            'val_losses': self.val_losses,
        }, path)

        ema_path = path.replace('.pth', '_ema.pth')
        self.ema.apply_shadow(self.model)
        torch.save({'model_state_dict': self.model.state_dict()}, ema_path)
        self.ema.restore(self.model)

        print(f"Checkpoint saved to {path}")
        print(f"EMA weights saved to {ema_path}")

    def plot_losses(self, save_path: Optional[str] = None):
        plt.figure(figsize=(8, 5))
        plt.plot(self.epoch_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Mamba Training Progress")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
