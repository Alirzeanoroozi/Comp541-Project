import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.trainer_utils import (
    attn_entropy,
    regression_metrics,
    plot_training_curves,
    plot_predictions,
    save_predictions_csv,
    update_metrics_summary_json,
)


class RegressionTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-5, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

        # Optional MIL entropy regularization (set from config externally)
        self.lam_entropy = 0.0

        # Early stopping
        self.early_stopping_patience = 20
        self.early_stopping_counter = 0
        self.best_val_loss_for_early_stop = float("inf")

        # Create save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_spearman": [],
            "val_spearman": [],
        }

        self.best_val_loss = float("inf")
        self.best_model_state = None

    # handles different batch types from unimodal/multimodal/LoRA dataloaders
    def _forward_batch(self, batch):
        # multimodal
        if isinstance(batch, (tuple, list)) and len(batch) == 10:
            (dna_pad, rna_pad, prot_pad, labels,
             dna_len, rna_len, prot_len,
             dna_mask, rna_mask, prot_mask) = batch

            dna_pad = dna_pad.to(self.device)
            rna_pad = rna_pad.to(self.device)
            prot_pad = prot_pad.to(self.device)

            dna_len = dna_len.to(self.device)
            rna_len = rna_len.to(self.device)
            prot_len = prot_len.to(self.device)

            targets = labels.to(self.device).float().view(-1)

            out = self.model(dna_pad, rna_pad, prot_pad, dna_len, rna_len, prot_len)

        # unimodal
        elif isinstance(batch, (tuple, list)) and len(batch) == 4:
            embeddings, labels, lengths, mask = batch

            embeddings = embeddings.to(self.device)
            mask = mask.to(self.device)
            targets = labels.to(self.device).float().view(-1)

            out = self.model(embeddings, mask=mask)

        # LoRA (raw sequences)
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            sequences, labels = batch
            targets = labels.to(self.device).float().view(-1)
            
            # Model expects sequences as a list/sequence of strings
            out = self.model(sequences)

        else:
            raise ValueError(
                f"Unexpected batch structure/length: type={type(batch)}, "
                f"len={len(batch) if isinstance(batch, (tuple, list)) else 'N/A'}"
            )

        # Model may optionally return (preds, aux)
        if isinstance(out, tuple):
            preds, aux = out
        else:
            preds, aux = out, {}

        preds = preds.float().view(-1)
        return preds, targets, aux

    def _compute_loss(self, preds, targets, aux):
        loss = self.criterion(preds, targets)

        # entropy regularization only for mil
        if self.lam_entropy > 0 and isinstance(aux, dict) and ("alpha" in aux):
            loss = loss + self.lam_entropy * attn_entropy(aux["alpha"])

        return loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n = 0
        all_preds = []
        all_targets = []

        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()

            preds, targets, aux = self._forward_batch(batch)
            loss = self._compute_loss(preds, targets, aux)

            loss.backward()
            self.optimizer.step()

            bs = targets.size(0)
            total_loss += loss.item() * bs
            n += bs

            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

        all_preds = np.asarray(all_preds)
        all_targets = np.asarray(all_targets)

        mse, rho = regression_metrics(all_targets, all_preds)
        avg_loss = total_loss / max(n, 1)

        return {
            "loss": float(avg_loss),
            "mse": float(mse),
            "spearman": float(rho),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        n = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                preds, targets, aux = self._forward_batch(batch)
                loss = self._compute_loss(preds, targets, aux)

                bs = targets.size(0)
                total_loss += loss.item() * bs
                n += bs

                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())

        all_preds = np.asarray(all_preds, dtype=np.float32)
        all_targets = np.asarray(all_targets, dtype=np.float32)

        mse, rho = regression_metrics(all_targets, all_preds)
        avg_loss = total_loss / max(n, 1)

        return {
            "loss": float(avg_loss),
            "mse": float(mse),
            "spearman": float(rho),
            "predictions": all_preds,
            "targets": all_targets,
        }

    def train(self, epochs):
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")

        for epoch in range(epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate(self.val_loader)

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_mse"].append(train_metrics["mse"])
            self.history["val_mse"].append(val_metrics["mse"])
            self.history["train_spearman"].append(train_metrics.get("spearman", 0.0))
            self.history["val_spearman"].append(val_metrics.get("spearman", 0.0))

            # Save best model (by val loss)
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(
                f"  Train Loss: {train_metrics['loss']:.4f}, "
                f"MSE: {train_metrics['mse']:.4f}, "
                f"Spearman: {train_metrics.get('spearman', 0.0):.4f}, "
                f"LR: {train_metrics['lr']:.6f}"
            )
            print(
                f"  Val Loss: {val_metrics['loss']:.4f}, "
                f"MSE: {val_metrics['mse']:.4f}, "
                f"Spearman: {val_metrics.get('spearman', 0.0):.4f}"
            )

            # Early stopping check
            if val_metrics["loss"] < self.best_val_loss_for_early_stop:
                self.best_val_loss_for_early_stop = val_metrics["loss"]
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                    print(f"Best validation loss: {self.best_val_loss_for_early_stop:.4f}")
                    break

            # Plot progress every 5 epochs or on last epoch
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                plot_training_curves(self.history, self.save_dir)
                plot_predictions(
                    val_metrics["predictions"],
                    val_metrics["targets"],
                    epoch=epoch + 1,
                    save_dir=self.save_dir,
                )

        # load best model weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")

        # Final plots + final val
        plot_training_curves(self.history, self.save_dir)

        final_val_metrics = self.validate(self.val_loader)
        plot_predictions(
            final_val_metrics["predictions"],
            final_val_metrics["targets"],
            epoch="final",
            save_dir=self.save_dir,
        )

        # Test evaluation
        test_metrics = self.validate(self.test_loader)
        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  MSE: {test_metrics['mse']:.4f}")
        print(f"  Spearman: {test_metrics.get('spearman', 0.0):.4f}")

        plot_predictions(
            test_metrics["predictions"],
            test_metrics["targets"],
            epoch="test",
            save_dir=self.save_dir,
        )

        # Save CSVs + metrics summary JSON
        self.save_values(final_val_metrics, epoch="final")
        self.save_values(test_metrics, epoch="test")

    def save_values(self, metrics, epoch):
        preds = metrics.get("predictions")
        targs = metrics.get("targets")
        if preds is not None and targs is not None:
            save_predictions_csv(preds, targs, self.save_dir / f"{epoch}_predictions_vs_targets.csv")

        run_metrics = {k: v for k, v in metrics.items() if k not in ("predictions", "targets")}
        run_metrics["run_key"] = str(epoch)

        update_metrics_summary_json(history=self.history, run_metrics=run_metrics, summary_path=self.save_dir / "metrics_summary.json")
