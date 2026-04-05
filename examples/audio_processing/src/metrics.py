"""
metrics.py
==========
Evaluation metrics for 3D Source Localization on Spatial LibriSpeech.

The primary metric is the **3D Angular Error** (geodesic distance on the
unit sphere), computed using the spherical law of cosines:

.. math::

    \\alpha = \\arccos\\!\\bigl(
        \\sin\\phi\\,\\sin\\hat{\\phi}
        + \\cos\\phi\\,\\cos\\hat{\\phi}\\,\\cos(\\theta - \\hat{\\theta})
    \\bigr)

where :math:`\\phi` is elevation and :math:`\\theta` is azimuth (both in
radians).  The formula gives the great-circle angle between two points on
the unit sphere, handling the wrap-around ambiguity that plagues naive
per-angle MAE.

The reported scalar is the **Median Absolute Error** (MedAE) of
:math:`\\alpha` across the evaluation set, expressed in **degrees**.
The median is preferred over the mean because it is robust to the
outlier predictions that untrained (or early-epoch) models produce.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def angular_error(
    azimuth_pred: Tensor,
    elevation_pred: Tensor,
    azimuth_true: Tensor,
    elevation_true: Tensor,
    eps: float = 1e-7,
) -> Tensor:
    """Per-sample 3D angular error (geodesic distance) in radians.

    Uses the spherical law of cosines:

    .. math::

        \\alpha = \\arccos\\!\\bigl(
            \\sin\\phi\\,\\sin\\hat{\\phi}
            + \\cos\\phi\\,\\cos\\hat{\\phi}\\,\\cos(\\theta - \\hat{\\theta})
        \\bigr)

    Parameters
    ----------
    azimuth_pred, elevation_pred : Tensor
        Predicted azimuth / elevation, shape ``(B,)``, radians.
    azimuth_true, elevation_true : Tensor
        Ground-truth azimuth / elevation, shape ``(B,)``, radians.
    eps : float
        Clamp tolerance for the ``arccos`` argument to avoid NaN at ±1.

    Returns
    -------
    Tensor
        Per-sample angular error, shape ``(B,)``, **radians**.
    """
    cos_angle = (
        torch.sin(elevation_pred) * torch.sin(elevation_true)
        + torch.cos(elevation_pred)
        * torch.cos(elevation_true)
        * torch.cos(azimuth_pred - azimuth_true)
    )
    cos_angle = cos_angle.clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos_angle)  # (B,) radians


def median_angular_error(
    azimuth_pred: Tensor,
    elevation_pred: Tensor,
    azimuth_true: Tensor,
    elevation_true: Tensor,
) -> float:
    """Median absolute 3D angular error across a batch, in **degrees**.

    Parameters
    ----------
    azimuth_pred, elevation_pred : Tensor
        Shape ``(B,)``, radians.
    azimuth_true, elevation_true : Tensor
        Shape ``(B,)``, radians.

    Returns
    -------
    float
        MedAE in degrees.
    """
    err_rad = angular_error(
        azimuth_pred, elevation_pred, azimuth_true, elevation_true
    )
    return float(torch.median(err_rad)) * (180.0 / math.pi)


# ---------------------------------------------------------------------------
# Stateful accumulator — collects predictions across batches then reduces
# ---------------------------------------------------------------------------

@dataclass
class AngularErrorAccumulator:
    """Accumulate per-sample angular errors across multiple batches.

    Typical usage inside an evaluation loop::

        acc = AngularErrorAccumulator()
        for batch in loader:
            features, az_true, el_true = batch
            az_pred, el_pred = model.predict(features)
            acc.update(az_pred, el_pred, az_true, el_true)

        results = acc.compute()
        print(results)  # {'median_deg': ..., 'mean_deg': ..., 'std_deg': ...}

    Parameters
    ----------
    device : str or torch.device, optional
        Device on which intermediate tensors are kept.  Defaults to CPU.
    """

    device: str = "cpu"
    _errors_rad: List[Tensor] = field(default_factory=list, repr=False)

    def reset(self) -> None:
        """Clear all accumulated errors."""
        self._errors_rad.clear()

    def update(
        self,
        azimuth_pred: Tensor,
        elevation_pred: Tensor,
        azimuth_true: Tensor,
        elevation_true: Tensor,
    ) -> None:
        """Accumulate per-sample errors from one batch.

        All tensors must have shape ``(B,)`` and be in **radians**.
        Tensors are moved to ``self.device`` automatically.
        """
        az_p = azimuth_pred.detach().to(self.device)
        el_p = elevation_pred.detach().to(self.device)
        az_t = azimuth_true.detach().to(self.device)
        el_t = elevation_true.detach().to(self.device)

        err = angular_error(az_p, el_p, az_t, el_t)  # (B,) radians
        self._errors_rad.append(err)

    def compute(self) -> Dict[str, float]:
        """Reduce accumulated errors to summary statistics.

        Returns
        -------
        dict with keys
            ``median_deg``, ``mean_deg``, ``std_deg``, ``n_samples``
        """
        if not self._errors_rad:
            raise RuntimeError("No samples accumulated. Call update() first.")

        all_errors = torch.cat(self._errors_rad)           # (N,) radians
        deg = all_errors * (180.0 / math.pi)

        return {
            "median_deg": float(torch.median(deg)),
            "mean_deg":   float(deg.mean()),
            "std_deg":    float(deg.std()),
            "n_samples":  int(all_errors.numel()),
        }

    def median_deg(self) -> float:
        """Convenience shortcut — returns just the median error in degrees."""
        return self.compute()["median_deg"]

    def __len__(self) -> int:
        return sum(e.numel() for e in self._errors_rad)


# ---------------------------------------------------------------------------
# Convenience wrapper for per-epoch logging
# ---------------------------------------------------------------------------

class AngularError:
    """High-level metric class for use in training / evaluation loops.

    Combines the :class:`AngularErrorAccumulator` with a per-epoch history
    so that training curves can be plotted after the run.

    Parameters
    ----------
    name : str
        Display name used in ``__repr__`` and history keys.
    device : str or torch.device
        Computation device.

    Example
    -------
    >>> metric = AngularError(name="val")
    >>> metric.reset()
    >>> metric.update(az_pred, el_pred, az_true, el_true)
    >>> score = metric.commit_epoch()      # appends to history, returns MedAE°
    >>> print(metric.history)              # [score]
    """

    def __init__(self, name: str = "angular_error", device: str = "cpu") -> None:
        self.name = name
        self.history: List[float] = []
        self._acc = AngularErrorAccumulator(device=device)

    # ------------------------------------------------------------------
    # Per-batch API (mirrors torchmetrics interface)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear per-epoch buffer (call at the start of each epoch)."""
        self._acc.reset()

    def update(
        self,
        azimuth_pred: Tensor,
        elevation_pred: Tensor,
        azimuth_true: Tensor,
        elevation_true: Tensor,
    ) -> None:
        """Accumulate one batch of predictions."""
        self._acc.update(azimuth_pred, elevation_pred, azimuth_true, elevation_true)

    def commit_epoch(self) -> float:
        """Compute MedAE°, append to history, reset buffer.

        Returns
        -------
        float
            Median angular error in degrees for this epoch.
        """
        score = self._acc.median_deg()
        self.history.append(score)
        self.reset()
        return score

    # ------------------------------------------------------------------
    # Direct computation (no state needed)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_batch(
        pred: Tensor,
        azimuth_true: Tensor,
        elevation_true: Tensor,
    ) -> float:
        """Compute MedAE° directly from a model output tensor.

        Parameters
        ----------
        pred : Tensor
            Shape ``(B, 2)`` — ``[:, 0]`` = azimuth, ``[:, 1]`` = elevation,
            radians.
        azimuth_true, elevation_true : Tensor
            Shape ``(B,)``, radians.

        Returns
        -------
        float
            Median angular error in degrees.
        """
        return median_angular_error(
            pred[:, 0], pred[:, 1], azimuth_true, elevation_true
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self.history)
        last = f"{self.history[-1]:.2f}°" if n else "—"
        return (
            f"AngularError(name={self.name!r}, "
            f"epochs={n}, last_MedAE={last})"
        )
