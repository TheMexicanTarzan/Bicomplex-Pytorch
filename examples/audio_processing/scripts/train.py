"""
train.py
========
Object-oriented training pipeline for 3D Source Localization.

Provides :class:`SpatialTrainer`, a self-contained training controller that
manages the full lifecycle for one model:

* ``train_epoch()``  — one forward + backward pass over the training set
* ``evaluate()``     — full validation pass returning the 3D angular error
* ``fit()``          — loop over N epochs, logging to stdout and recording
                       training curves for later plotting

Design choices
--------------
* **Loss**: mean squared error on raw radian outputs (``nn.MSELoss``).
  MSE penalises large angular deviations more than MAE, matching our goal
  of minimising gross mis-localisation.
* **Optimiser**: AdamW, lr=1e-5, weight_decay=0.01.  The small learning
  rate suits fine-grained regression on a moderately large dataset;
  weight decay regularises the large MLP head.
* **Scheduler**: CosineAnnealingLR that decays lr to ``eta_min=1e-7`` over
  the full training run, providing a smooth warm-to-cool schedule without
  requiring manual step-size tuning.
* **Gradient clipping**: global norm clipped to 1.0 to stabilise early
  training when bicomplex weights are near random initialisation.
* **Device handling**: the trainer moves the model and batches to
  ``device`` automatically; all metric computation stays on CPU.

Typical usage
-------------
>>> trainer = SpatialTrainer(
...     model=bicomplex_model,
...     train_loader=train_loader,
...     val_loader=val_loader,
...     device="cuda",
... )
>>> history = trainer.fit(n_epochs=20)
>>> trainer.plot_history()
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Local
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import AngularError

logger = logging.getLogger(__name__)


# ============================================================================
# Training history container
# ============================================================================

@dataclass
class TrainingHistory:
    """Stores per-epoch training and validation curves.

    Attributes
    ----------
    train_loss : list of float
        Mean MSE loss per training epoch.
    val_median_deg : list of float
        Validation 3D angular error (MedAE°) per epoch.
    learning_rates : list of float
        Learning rate recorded at the end of each epoch.
    epoch_times_s : list of float
        Wall-clock time per epoch in seconds.
    """

    model_name:      str        = "model"
    train_loss:      List[float] = field(default_factory=list)
    val_median_deg:  List[float] = field(default_factory=list)
    learning_rates:  List[float] = field(default_factory=list)
    epoch_times_s:   List[float] = field(default_factory=list)

    @property
    def n_epochs(self) -> int:
        return len(self.train_loss)

    @property
    def best_val_deg(self) -> float:
        """Best (lowest) validation MedAE° recorded so far."""
        return min(self.val_median_deg) if self.val_median_deg else float("inf")

    @property
    def best_epoch(self) -> int:
        """0-based epoch index with the best validation score."""
        return int(torch.tensor(self.val_median_deg).argmin().item())

    def __repr__(self) -> str:
        return (
            f"TrainingHistory(model={self.model_name!r}, "
            f"epochs={self.n_epochs}, "
            f"best_val={self.best_val_deg:.2f}° @ epoch {self.best_epoch + 1})"
        )


# ============================================================================
# SpatialTrainer
# ============================================================================

class SpatialTrainer:
    """Self-contained training controller for one Source-Localization model.

    Parameters
    ----------
    model : nn.Module
        The model to train.  Must accept ``(B, 6, F, T)`` input tensors
        and output ``(B, 2)`` ``(azimuth, elevation)`` in radians.
    train_loader : DataLoader
        Yields ``(features, azimuth, elevation)`` tuples with
        ``features`` shape ``(B, 6, F, T)``.
    val_loader : DataLoader
        Same format as ``train_loader``.
    device : str or torch.device
        Target compute device.  Default ``"cpu"``.
    lr : float
        Initial AdamW learning rate.  Default ``1e-5``.
    weight_decay : float
        AdamW weight decay (L2 regularisation).  Default ``0.01``.
    grad_clip_norm : float
        Maximum global gradient norm for clipping.  Default ``1.0``.
    n_epochs : int
        Total training epochs (used to configure the cosine scheduler).
        Default ``20``.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str | torch.device = "cpu",
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        grad_clip_norm: float = 1.0,
        n_epochs: int = 20,
    ) -> None:
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = torch.device(device)
        self.grad_clip_norm = grad_clip_norm
        self.n_epochs       = n_epochs

        # Move model to device
        self.model.to(self.device)

        # ── Loss ──────────────────────────────────────────────────────────
        self.criterion = nn.MSELoss()

        # ── Optimiser: AdamW ──────────────────────────────────────────────
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # ── Cosine annealing scheduler ────────────────────────────────────
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=n_epochs,
            eta_min=1e-7,
        )

        # ── Metric ────────────────────────────────────────────────────────
        self._val_metric = AngularError(name="val", device="cpu")

        # ── History ───────────────────────────────────────────────────────
        model_name = getattr(model, "__class__", type(model)).__name__
        self.history = TrainingHistory(model_name=model_name)

    # ------------------------------------------------------------------
    # Single-epoch training
    # ------------------------------------------------------------------

    def train_epoch(self) -> float:
        """Run one full pass over the training set.

        Returns
        -------
        float
            Mean MSE loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for features, az_true, el_true in self.train_loader:
            # ── Move to device ────────────────────────────────────────────
            features = features.to(self.device)          # (B, 6, F, T)
            targets  = torch.stack([az_true, el_true], dim=1).to(self.device)  # (B, 2)

            # ── Forward ───────────────────────────────────────────────────
            self.optimizer.zero_grad(set_to_none=True)
            preds = self.model(features)                 # (B, 2)

            # ── Loss + backward ───────────────────────────────────────────
            loss = self.criterion(preds, targets)
            loss.backward()

            # ── Gradient clipping ─────────────────────────────────────────
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip_norm
            )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set.

        Returns
        -------
        dict
            Keys: ``median_deg``, ``mean_deg``, ``std_deg``, ``n_samples``.
        """
        self.model.eval()
        self._val_metric.reset()

        with torch.no_grad():
            for features, az_true, el_true in self.val_loader:
                features = features.to(self.device)
                preds    = self.model(features)          # (B, 2)

                # Bring predictions back to CPU for metric
                self._val_metric.update(
                    azimuth_pred   = preds[:, 0].cpu(),
                    elevation_pred = preds[:, 1].cpu(),
                    azimuth_true   = az_true,
                    elevation_true = el_true,
                )

        return self._val_metric._acc.compute()

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self, n_epochs: Optional[int] = None) -> TrainingHistory:
        """Train for *n_epochs* epochs, logging progress each epoch.

        Parameters
        ----------
        n_epochs : int, optional
            Override the ``n_epochs`` passed to ``__init__``.

        Returns
        -------
        TrainingHistory
            Object containing train loss and validation MedAE° curves.
        """
        epochs = n_epochs if n_epochs is not None else self.n_epochs
        model_name = self.history.model_name

        print(f"\n{'═' * 65}")
        print(f"  Training: {model_name}")
        print(f"  Epochs: {epochs}  |  "
              f"Device: {self.device}  |  "
              f"LR: {self.optimizer.param_groups[0]['lr']:.1e}")
        print(f"{'═' * 65}")
        print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MedAE°':>11}  "
              f"{'LR':>9}  {'Time':>6}")
        print(f"{'-' * 65}")

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()

            # ── Train ─────────────────────────────────────────────────────
            train_loss = self.train_epoch()

            # ── Validate ──────────────────────────────────────────────────
            val_stats  = self.evaluate()
            val_med    = val_stats["median_deg"]

            # ── Scheduler step (after optimizer step) ─────────────────────
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.perf_counter() - t0

            # ── Record history ────────────────────────────────────────────
            self.history.train_loss.append(train_loss)
            self.history.val_median_deg.append(val_med)
            self.history.learning_rates.append(current_lr)
            self.history.epoch_times_s.append(elapsed)

            # ── Log to stdout ─────────────────────────────────────────────
            marker = " ◀ best" if val_med <= self.history.best_val_deg else ""
            print(
                f"{epoch:>6}  {train_loss:>10.5f}  {val_med:>10.2f}°  "
                f"{current_lr:>9.2e}  {elapsed:>5.1f}s{marker}"
            )

            logger.info(
                "%s | epoch %d/%d | loss=%.5f | val_MedAE=%.2f° | lr=%.2e",
                model_name, epoch, epochs, train_loss, val_med, current_lr,
            )

        print(f"{'═' * 65}")
        print(f"  Best val MedAE: {self.history.best_val_deg:.2f}° "
              f"(epoch {self.history.best_epoch + 1})")
        print(f"{'═' * 65}\n")

        return self.history

    # ------------------------------------------------------------------
    # Plotting helper
    # ------------------------------------------------------------------

    def plot_history(
        self,
        ax_loss=None,
        ax_val=None,
        label: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        """Plot training loss and validation MedAE° curves onto provided axes.

        If axes are not provided, a new figure is created.

        Parameters
        ----------
        ax_loss, ax_val : matplotlib Axes, optional
            Pre-created axes (for overlaying multiple models).
        label : str, optional
            Legend label; defaults to ``history.model_name``.
        color : str, optional
            Line colour; defaults to matplotlib's automatic cycling.
        """
        import matplotlib.pyplot as plt

        own_fig = ax_loss is None
        if own_fig:
            fig, (ax_loss, ax_val) = plt.subplots(1, 2, figsize=(12, 4))

        lbl = label or self.history.model_name
        epochs = range(1, self.history.n_epochs + 1)

        ax_loss.plot(epochs, self.history.train_loss,
                     label=lbl, color=color, linewidth=1.5)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("MSE loss")
        ax_loss.set_title("Training loss")
        ax_loss.legend(fontsize=8)

        ax_val.plot(epochs, self.history.val_median_deg,
                    label=lbl, color=color, linewidth=1.5)
        ax_val.set_xlabel("Epoch")
        ax_val.set_ylabel("Median angular error (°)")
        ax_val.set_title("Validation 3D angular error")
        ax_val.legend(fontsize=8)

        if own_fig:
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SpatialTrainer("
            f"model={self.history.model_name}, "
            f"device={self.device}, "
            f"lr={self.optimizer.param_groups[0]['lr']:.1e}, "
            f"wd={self.optimizer.param_groups[0]['weight_decay']:.3f}, "
            f"epochs_done={self.history.n_epochs}"
            f")"
        )
