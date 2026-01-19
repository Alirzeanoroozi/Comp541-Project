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

# classification metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


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

    def _is_fusion_mil(self) -> bool:
        """
        Best-effort check to detect whether the *model* is using FusionMIL and returns alpha.
        We don't require config plumbing; we just check the aux dict at runtime.
        """
        return True  # used only as a semantic helper; runtime checks use aux.get("alpha")

    def _maybe_stack_alpha(self, alpha_list):
        if not alpha_list:
            return None
        try:
            return torch.cat(alpha_list, dim=0)  # [N, 3] expected
        except Exception:
            return None

    def _save_test_alpha(self, alpha_tensor, filename="test_alpha.pt"):
        if alpha_tensor is None:
            return
        # store raw per-sample alphas so you can compute means/plots later
        torch.save(alpha_tensor.detach().cpu(), self.save_dir / filename)
        # also store a quick summary for convenience
        mean = alpha_tensor.detach().cpu().mean(dim=0)
        torch.save(mean, self.save_dir / "test_alpha_mean.pt")

    # handles different batch types from unimodal/multimodal dataloaders
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

    def validate(self, loader, collect_alpha: bool = False):
        self.model.eval()
        total_loss = 0.0
        n = 0
        all_preds = []
        all_targets = []
        all_alpha = []  # list of [B,3]

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                preds, targets, aux = self._forward_batch(batch)
                loss = self._compute_loss(preds, targets, aux)

                if collect_alpha and isinstance(aux, dict) and ("alpha" in aux) and (aux["alpha"] is not None):
                    a = aux["alpha"]
                    if isinstance(a, torch.Tensor) and a.dim() == 2:
                        all_alpha.append(a.detach().cpu())

                bs = targets.size(0)
                total_loss += loss.item() * bs
                n += bs

                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())

        all_preds = np.asarray(all_preds, dtype=np.float32)
        all_targets = np.asarray(all_targets, dtype=np.float32)

        mse, rho = regression_metrics(all_targets, all_preds)
        avg_loss = total_loss / max(n, 1)

        out = {
            "loss": float(avg_loss),
            "mse": float(mse),
            "spearman": float(rho),
            "predictions": all_preds,
            "targets": all_targets,
        }

        if collect_alpha:
            alpha_tensor = self._maybe_stack_alpha(all_alpha)
            out["alpha"] = alpha_tensor  # torch.Tensor or None

        return out

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

        # Test evaluation (+ collect alpha if model provides it, i.e., fusion_mil)
        test_metrics = self.validate(self.test_loader, collect_alpha=True)
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

        # If alpha was collected, save it
        if "alpha" in test_metrics and test_metrics["alpha"] is not None:
            self._save_test_alpha(test_metrics["alpha"])

        # Save CSVs + metrics summary JSON
        self.save_values(final_val_metrics, epoch="final")
        self.save_values(test_metrics, epoch="test")

    def save_values(self, metrics, epoch):
        preds = metrics.get("predictions")
        targs = metrics.get("targets")
        if preds is not None and targs is not None:
            save_predictions_csv(preds, targs, self.save_dir / f"{epoch}_predictions_vs_targets.csv")

        # do not include raw tensors in summary json
        run_metrics = {k: v for k, v in metrics.items() if k not in ("predictions", "targets", "alpha")}
        run_metrics["run_key"] = str(epoch)

        update_metrics_summary_json(
            history=self.history,
            run_metrics=run_metrics,
            summary_path=self.save_dir / "metrics_summary.json",
        )



class ClassificationTrainer:

    def __init__(self, model, train_loader, val_loader, test_loader, device, save_dir,
                 num_classes=3, lr=3e-5, weight_decay=1e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = int(num_classes)

        # Optional MIL entropy regularization (set from config externally)
        self.lam_entropy = 0.0

        # Save dir
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Loss (multiclass)
        self.criterion = nn.CrossEntropyLoss()

        # If model outputs [B] or [B,1], we use this to create [B,C]
        self.logit_proj = None  # created lazily on first batch if needed

        # Optimizer (note: include proj params once created)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self._optim_lr = lr
        self._optim_wd = weight_decay

        # Early stopping
        self.early_stopping_patience = 20
        self.early_stopping_counter = 0
        self.best_val_loss_for_early_stop = float("inf")

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "val_f1": [],
            "train_auc": [],
            "val_auc": [],
        }

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.best_proj_state = None

    def _maybe_stack_alpha(self, alpha_list):
        if not alpha_list:
            return None
        try:
            return torch.cat(alpha_list, dim=0)
        except Exception:
            return None

    def _save_test_alpha(self, alpha_tensor, filename="test_alpha.pt"):
        if alpha_tensor is None:
            return
        torch.save(alpha_tensor.detach().cpu(), self.save_dir / filename)
        mean = alpha_tensor.detach().cpu().mean(dim=0)
        torch.save(mean, self.save_dir / "test_alpha_mean.pt")

    def _ensure_proj(self):
        """
        Ensure projection head exists and optimizer includes it.
        Called when we detect scalar logits.
        """
        if self.logit_proj is None:
            self.logit_proj = nn.Linear(1, self.num_classes).to(self.device)

            # Rebuild optimizer to include projection parameters too
            params = list(self.model.parameters()) + list(self.logit_proj.parameters())
            self.optimizer = optim.Adam(params, lr=self._optim_lr, weight_decay=self._optim_wd)

    def _coerce_logits_to_multiclass(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Returns logits shaped [B, num_classes] for multiclass CE.
        - If logits are [B, C], uses them.
        - If logits are [B] or [B,1], projects them to [B, C].
        """
        if logits.dim() == 2 and logits.size(1) == self.num_classes:
            return logits

        if logits.dim() == 1:
            self._ensure_proj()
            return self.logit_proj(logits.view(-1, 1))

        if logits.dim() == 2 and logits.size(1) == 1:
            self._ensure_proj()
            return self.logit_proj(logits)

        raise ValueError(
            f"Model returned logits with shape {tuple(logits.shape)}; "
            f"expected [B,{self.num_classes}] or [B] or [B,1] for multiclass coercion."
        )

    # handles different batch types from unimodal/multimodal dataloaders
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

            targets = labels.to(self.device).long().view(-1)
            out = self.model(dna_pad, rna_pad, prot_pad, dna_len, rna_len, prot_len)

        # unimodal
        elif isinstance(batch, (tuple, list)) and len(batch) == 4:
            embeddings, labels, lengths, mask = batch
            embeddings = embeddings.to(self.device)
            mask = mask.to(self.device)
            targets = labels.to(self.device).long().view(-1)
            out = self.model(embeddings, mask=mask)

        else:
            raise ValueError(
                f"Unexpected batch structure/length: type={type(batch)}, "
                f"len={len(batch) if isinstance(batch, (tuple, list)) else 'N/A'}"
            )

        if isinstance(out, tuple):
            logits, aux = out
        else:
            logits, aux = out, {}

        logits = self._coerce_logits_to_multiclass(logits)
        return logits, targets, aux

    def _compute_loss(self, logits, targets, aux):
        loss = self.criterion(logits, targets)

        if self.lam_entropy > 0 and isinstance(aux, dict) and ("alpha" in aux):
            loss = loss + self.lam_entropy * attn_entropy(aux["alpha"])

        return loss

    def _compute_metrics(self, y_true_t, logits_t):
        y_true = y_true_t.detach().cpu().numpy().astype(int)
        probs = torch.softmax(logits_t, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)

        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, average="macro"))

        auc = None
        try:
            if self.num_classes == 2:
                auc = float(roc_auc_score(y_true, probs[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        except ValueError:
            auc = None

        return acc, f1, auc, preds, probs

    def train_epoch(self):
        self.model.train()
        if self.logit_proj is not None:
            self.logit_proj.train()

        total_loss = 0.0
        n = 0

        all_logits = []
        all_targets = []

        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()

            logits, targets, aux = self._forward_batch(batch)
            loss = self._compute_loss(logits, targets, aux)

            loss.backward()
            self.optimizer.step()

            bs = targets.size(0)
            total_loss += float(loss.item()) * bs
            n += bs

            all_logits.append(logits.detach())
            all_targets.append(targets.detach())

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        acc, f1, auc, _, _ = self._compute_metrics(all_targets, all_logits)
        avg_loss = total_loss / max(n, 1)

        return {
            "loss": float(avg_loss),
            "acc": float(acc),
            "f1": float(f1),
            "auc": None if auc is None else float(auc),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def validate(self, loader, collect_alpha: bool = False):
        self.model.eval()
        if self.logit_proj is not None:
            self.logit_proj.eval()

        total_loss = 0.0
        n = 0

        all_logits = []
        all_targets = []
        all_alpha = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                logits, targets, aux = self._forward_batch(batch)
                loss = self._compute_loss(logits, targets, aux)

                if collect_alpha and isinstance(aux, dict) and ("alpha" in aux) and (aux["alpha"] is not None):
                    a = aux["alpha"]
                    if isinstance(a, torch.Tensor) and a.dim() == 2:
                        all_alpha.append(a.detach().cpu())

                bs = targets.size(0)
                total_loss += float(loss.item()) * bs
                n += bs

                all_logits.append(logits)
                all_targets.append(targets)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        acc, f1, auc, preds, probs = self._compute_metrics(all_targets, all_logits)
        avg_loss = total_loss / max(n, 1)

        out = {
            "loss": float(avg_loss),
            "acc": float(acc),
            "f1": float(f1),
            "auc": None if auc is None else float(auc),
            "predictions": preds,
            "probs": probs,
            "targets": all_targets.detach().cpu().numpy(),
        }

        if collect_alpha:
            out["alpha"] = self._maybe_stack_alpha(all_alpha)

        return out

    def train(self, epochs):
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")
        print(f"Num classes: {self.num_classes}")

        for epoch in range(epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate(self.val_loader)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_acc"].append(val_metrics["acc"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["train_auc"].append(train_metrics["auc"] if train_metrics["auc"] is not None else np.nan)
            self.history["val_auc"].append(val_metrics["auc"] if val_metrics["auc"] is not None else np.nan)

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.best_proj_state = None if self.logit_proj is None else {k: v.detach().cpu().clone() for k, v in self.logit_proj.state_dict().items()}

            print(f"Epoch {epoch+1}/{epochs}")
            print(
                f"  Train Loss: {train_metrics['loss']:.4f}, "
                f"Acc: {train_metrics['acc']:.4f}, "
                f"F1: {train_metrics['f1']:.4f}, "
                f"AUC: {train_metrics['auc'] if train_metrics['auc'] is not None else 'NA'}, "
                f"LR: {train_metrics['lr']:.6f}"
            )
            print(
                f"  Val Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['acc']:.4f}, "
                f"F1: {val_metrics['f1']:.4f}, "
                f"AUC: {val_metrics['auc'] if val_metrics['auc'] is not None else 'NA'}"
            )

            if val_metrics["loss"] < self.best_val_loss_for_early_stop:
                self.best_val_loss_for_early_stop = val_metrics["loss"]
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                    print(f"Best validation loss: {self.best_val_loss_for_early_stop:.4f}")
                    break

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                plot_training_curves(self.history, self.save_dir)

        # Restore best weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        if self.logit_proj is not None and self.best_proj_state is not None:
            self.logit_proj.load_state_dict(self.best_proj_state)

        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")

        plot_training_curves(self.history, self.save_dir)

        final_val_metrics = self.validate(self.val_loader)
        test_metrics = self.validate(self.test_loader, collect_alpha=True)

        print(f"\nTest Results:")
        print(
            f"  Loss: {test_metrics['loss']:.4f}\n"
            f"  Acc:  {test_metrics['acc']:.4f}\n"
            f"  F1:   {test_metrics['f1']:.4f}\n"
            f"  AUC:  {test_metrics['auc'] if test_metrics['auc'] is not None else 'NA'}"
        )

        if "alpha" in test_metrics and test_metrics["alpha"] is not None:
            self._save_test_alpha(test_metrics["alpha"])

        self.save_values(final_val_metrics, epoch="final")
        self.save_values(test_metrics, epoch="test")

    def save_values(self, metrics, epoch):
        preds = metrics.get("predictions")
        targs = metrics.get("targets")
        if preds is not None and targs is not None:
            save_predictions_csv(preds, targs, self.save_dir / f"{epoch}_predictions_vs_targets.csv")

        run_metrics = {k: v for k, v in metrics.items() if k not in ("predictions", "targets", "probs", "alpha")}
        run_metrics["run_key"] = str(epoch)

        update_metrics_summary_json(
            history=self.history,
            run_metrics=run_metrics,
            summary_path=self.save_dir / "metrics_summary.json",
        )


        update_metrics_summary_json(history=self.history, run_metrics=run_metrics, summary_path=self.save_dir / "metrics_summary.json")
