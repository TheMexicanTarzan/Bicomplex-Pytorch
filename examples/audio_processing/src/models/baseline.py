"""
SpatialBaselineNet
==================
Real-valued 3D-CNN baseline for 3D Source Localization on Spatial LibriSpeech.

The network processes **active** and **reactive** acoustic intensity features
(produced by :class:`ActiveReactivePreprocessor`) through two independent,
parallel 3D-convolutional branches.  The learned representations are fused
only at the final MLP stage to predict azimuth and elevation.

Architecture overview
---------------------
::

    Input (B, 6, F, T)
        │
        ├── Active  (B, 1, 3, F, T)           Reactive (B, 1, 3, F, T)
        │       │                                    │
        │   Conv3d 1→2 ─ BN ─ ELU ─ MaxPool     Conv3d 1→2 ─ BN ─ ELU ─ MaxPool
        │   Conv3d 2→4 ─ BN ─ ELU ─ MaxPool     Conv3d 2→4 ─ BN ─ ELU ─ MaxPool
        │   Conv3d 4→8 ─ BN ─ ELU ─ MaxPool     Conv3d 4→8 ─ BN ─ ELU ─ MaxPool
        │   Conv3d 8→16 ─ BN ─ ELU ─ MaxPool    Conv3d 8→16 ─ BN ─ ELU ─ MaxPool
        │   Dropout(0.5)                         Dropout(0.5)
        │       │                                    │
        │       └────── Flatten ─── Concatenate ─────┘
        │                               │
        │                    Linear → ELU
        │                    Linear → ELU
        │                    Linear → 2  (azimuth, elevation)
        │
        └── Output (B, 2)

All convolutional weights are initialised with **Kaiming uniform**.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class _ConvBranch(nn.Module):
    """A single 3D-convolutional branch (active *or* reactive).

    Four Conv3d blocks, each followed by BatchNorm3d → ELU → MaxPool3d,
    plus a final 50 % Dropout.

    Parameters
    ----------
    in_channels:
        Number of input channels for the first Conv3d layer.  Typically 1,
        since the 3 directional components are arranged along the depth
        dimension rather than the channel dimension.
    channel_depths:
        Sequence of output channels for each of the four Conv3d layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channel_depths: Tuple[int, ...] = (2, 4, 8, 16),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channel_depths:
            layers.extend([
                nn.Conv3d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm3d(c_out),
                nn.ELU(inplace=True),
                # Pool only in frequency (H) and time (W); keep depth (D=3)
                nn.MaxPool3d(kernel_size=(1, 2, 2)),
            ])
            c_in = c_out

        layers.append(nn.Dropout(p=0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape ``(B, C_in, D, F, T)``.

        Returns
        -------
        Tensor
            Shape ``(B, C_out, D', F', T')`` after four pool stages.
        """
        return self.net(x)


class SpatialBaselineNet(nn.Module):
    """Real-valued dual-branch 3D-CNN for 3D Source Localization.

    The model expects input features from
    :class:`~audio_processing.src.preprocess.ActiveReactivePreprocessor`:
    a tensor of shape ``(B, 6, F, T_frames)`` where channels 0-2 are
    active intensity ``(Ia_x, Ia_y, Ia_z)`` and channels 3-5 are reactive
    intensity ``(Ir_x, Ir_y, Ir_z)``.

    Internally, the 6-channel input is split and reshaped into two
    independent 5-D volumes:

    * **Active branch**:  ``(B, 1, 3, F, T)`` — 1 input channel, depth=3
    * **Reactive branch**: ``(B, 1, 3, F, T)`` — same layout

    Each branch runs through four Conv3d-BN-ELU-MaxPool blocks with channel
    depths [2, 4, 8, 16].  The flattened outputs are concatenated and fed
    into a 3-layer MLP that produces the final 2-node output (azimuth,
    elevation in radians).

    Parameters
    ----------
    n_freq:
        Number of STFT frequency bins (``n_fft // 2 + 1``).
        Default 257 for ``n_fft=512``.
    n_time:
        Number of STFT time frames per chunk.
        Default 63 for ``chunk_duration=0.5s``, ``hop_length=128``,
        ``center=True``, ``sample_rate=16_000``.
    mlp_hidden_1:
        Number of units in the first MLP hidden layer.  Default 128.
    mlp_hidden_2:
        Number of units in the second MLP hidden layer.  Default 64.
    """

    # Channel indices in the preprocessor output
    _ACTIVE_SLICE = slice(0, 3)   # Ia_x, Ia_y, Ia_z
    _REACTIVE_SLICE = slice(3, 6) # Ir_x, Ir_y, Ir_z

    def __init__(
        self,
        n_freq: int = 257,
        n_time: int = 63,
        mlp_hidden_1: int = 128,
        mlp_hidden_2: int = 64,
    ) -> None:
        super().__init__()

        self.n_freq = n_freq
        self.n_time = n_time

        # ── Parallel 3D-CNN branches ──────────────────────────────────────
        self.active_branch = _ConvBranch(in_channels=1,
                                         channel_depths=(2, 4, 8, 16))
        self.reactive_branch = _ConvBranch(in_channels=1,
                                           channel_depths=(2, 4, 8, 16))

        # ── Compute the flattened feature dimension dynamically ───────────
        # Run a dummy tensor through one branch to determine the size.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 3, n_freq, n_time)
            dummy_out = self.active_branch(dummy)
            flat_per_branch = dummy_out.numel()   # total elements for B=1

        concat_dim = 2 * flat_per_branch

        # ── 3-layer MLP (fusion head) ────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, mlp_hidden_1),
            nn.ELU(inplace=True),
            nn.Linear(mlp_hidden_1, mlp_hidden_2),
            nn.ELU(inplace=True),
            nn.Linear(mlp_hidden_2, 2),  # azimuth, elevation
        )

        # ── Weight initialisation ─────────────────────────────────────────
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply Kaiming uniform initialisation to all Conv3d and Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape ``(B, 6, F, T)`` — 6-channel active/reactive intensity
            features from ``ActiveReactivePreprocessor``.

        Returns
        -------
        Tensor
            Shape ``(B, 2)`` — predicted ``(azimuth, elevation)`` in radians.
        """
        # ── Split into active (channels 0-2) and reactive (channels 3-5) ─
        active = x[:, self._ACTIVE_SLICE, :, :]    # (B, 3, F, T)
        reactive = x[:, self._REACTIVE_SLICE, :, :] # (B, 3, F, T)

        # ── Reshape for Conv3d: (B, 3, F, T) → (B, 1, 3, F, T) ──────────
        # The 3 directional components become the depth dimension; a single
        # input channel lets the 3D kernels jointly convolve across
        # direction × frequency × time.
        active = active.unsqueeze(1)    # (B, 1, 3, F, T)
        reactive = reactive.unsqueeze(1) # (B, 1, 3, F, T)

        # ── Parallel branches ─────────────────────────────────────────────
        h_active = self.active_branch(active)     # (B, 16, D', F', T')
        h_reactive = self.reactive_branch(reactive) # (B, 16, D', F', T')

        # ── Flatten and concatenate ───────────────────────────────────────
        h_active = h_active.flatten(start_dim=1)    # (B, flat)
        h_reactive = h_reactive.flatten(start_dim=1) # (B, flat)
        h = torch.cat([h_active, h_reactive], dim=1)  # (B, 2*flat)

        # ── MLP fusion head ───────────────────────────────────────────────
        out = self.mlp(h)  # (B, 2)
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Convenience wrapper that returns separate azimuth and elevation.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, 6, F, T)``.

        Returns
        -------
        azimuth : Tensor
            Shape ``(B,)`` — predicted azimuth in radians.
        elevation : Tensor
            Shape ``(B,)`` — predicted elevation in radians.
        """
        out = self.forward(x)
        return out[:, 0], out[:, 1]

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        base = super().__repr__()
        return (
            f"{base}\n"
            f"  Trainable parameters: {self.count_parameters():,}"
        )
