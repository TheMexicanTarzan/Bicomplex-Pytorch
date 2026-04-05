"""
ComplexSpatialNet & BicomplexSpatialNet
=======================================
Complex- and bicomplex-valued CNNs for 3D Source Localization on Spatial
LibriSpeech.

These models replace the dual-branch architecture of
:class:`SpatialBaselineNet` with a **unified convolutional trunk** that
encodes inter-channel relationships algebraically rather than via
independent parallel streams.

Hierarchy of representations
-----------------------------
::

    Real baseline  →  (Ia_x, Ia_y, Ia_z)  |  (Ir_x, Ir_y, Ir_z)
                      two independent Conv3d branches, fused at MLP

    Complex        →  z_d = Ia_d + j·Ir_d  for d ∈ {x, y, z}
                      3 complex channels, single Conv2d trunk
                      couples active ↔ reactive per direction axis

    Bicomplex      →  q₀ = Ia_x + Ir_x·i₁ + Ia_y·i₂ + Ir_y·i₁i₂
                      q₁ = Ia_z + Ir_z·i₁ + 0·i₂   + 0·i₁i₂
                      2 bicomplex channels, single Conv2d trunk
                      couples active ↔ reactive AND x-axis ↔ y-axis

Input / Output
--------------
Both models accept the same ``(B, 6, F, T)`` feature tensor produced by
:class:`ActiveReactivePreprocessor` (channels 0-2 = Ia_x/y/z, channels
3-5 = Ir_x/y/z) and output ``(B, 2)`` — azimuth and elevation in radians.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch import Tensor

# ── Bicomplex-Pytorch library ────────────────────────────────────────────────
from bicomplex_pytorch.nn.modules.conv import ComplexConv2d, BiComplexConv2d


# ============================================================================
# Shared utility
# ============================================================================

def _pooled_dims(n_freq: int, n_time: int, n_pools: int) -> Tuple[int, int]:
    """Return (F, T) after *n_pools* rounds of MaxPool2d(kernel=2)."""
    f, t = n_freq, n_time
    for _ in range(n_pools):
        f //= 2
        t //= 2
    return f, t


def _init_mlp(module: nn.Module) -> None:
    """Kaiming uniform init on all real Linear layers in *module*."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ============================================================================
# Private building blocks
# ============================================================================

class _ComplexConvBlock(nn.Module):
    """ComplexConv2d → BN(real) + BN(imag) → ELU → MaxPool2d.

    Batch-norm and activation are applied independently to the real and
    imaginary parts — the standard practice when dedicated complex
    non-linearities are unavailable.  The algebraic coupling of real and
    imaginary components is provided by :class:`ComplexConv2d` itself via
    the complex multiplication rule  ``(a+jb)(c+jd) = (ac-bd)+j(ad+bc)``.

    Parameters
    ----------
    in_channels, out_channels:
        Number of complex input / output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ComplexConv2d(
            in_channels, out_channels, kernel_size=3, padding=1,
        )
        self.bn_real = nn.BatchNorm2d(out_channels)
        self.bn_imag = nn.BatchNorm2d(out_channels)
        self.pool    = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, C_in, F, T)`` complex → ``(B, C_out, F//2, T//2)`` complex."""
        x = self.conv(x)
        x = torch.complex(self.bn_real(x.real), self.bn_imag(x.imag))
        x = torch.complex(F_torch.elu(x.real), F_torch.elu(x.imag))
        x = torch.complex(self.pool(x.real), self.pool(x.imag))
        return x


class _BiComplexConvBlock(nn.Module):
    """BiComplexConv2d (standard-format) → BN → ELU → MaxPool2d.

    In **standard format** the bicomplex tensor is a plain real tensor with
    ``C_out × 4`` channels arranged as
    ``[a₁…aₙ, b₁…bₙ, c₁…cₙ, d₁…dₙ]``.  All post-convolution operations
    run directly on this real tensor; the bicomplex algebra is enforced
    inside :class:`BiComplexConv2d` via the idempotent decomposition.

    Parameters
    ----------
    in_channels, out_channels:
        Number of **bicomplex** input / output channels (not ×4).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = BiComplexConv2d(
            in_channels, out_channels, kernel_size=3, padding=1,
            input_format="standard", output_format="standard",
        )
        # Standard format → C_out*4 real channels
        self.bn   = nn.BatchNorm2d(out_channels * 4)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, C_in×4, F, T)`` real → ``(B, C_out×4, F//2, T//2)`` real."""
        x = self.conv(x)
        x = self.bn(x)
        x = F_torch.elu(x)
        x = self.pool(x)
        return x


# ============================================================================
# ComplexSpatialNet
# ============================================================================

class ComplexSpatialNet(nn.Module):
    """Complex-valued unified CNN for 3D Source Localization.

    **Encoding**: The 6 real intensity channels are paired per direction
    axis into 3 complex channels:

    .. math::

        z_d = I_{a,d} + j\\,I_{r,d}, \\quad d \\in \\{x,\\,y,\\,z\\}

    The real part carries active intensity (net power flow) and the
    imaginary part carries reactive intensity (stored/circulating energy).
    This mirrors the original physics: both quantities are the real and
    imaginary parts of the same cross-spectrum ``W(f,t)·Xd(f,t)*``.

    **Trunk**: Four :class:`_ComplexConvBlock` stages, channel depths
    ``(2, 4, 8, 16)`` (matching the baseline).

    **Head**: Complex feature maps are decoded to real by concatenating
    real ∥ imaginary parts, then passed through Dropout(0.5), flattened,
    and fed into a 3-layer MLP → ``(azimuth, elevation)``.

    Parameters
    ----------
    n_freq : int
        STFT frequency bins ``F``.  Default 257 (``n_fft=512``).
    n_time : int
        STFT frames per 0.5 s chunk ``T``.  Default 63.
    mlp_hidden_1, mlp_hidden_2 : int
        MLP hidden-layer widths.  Defaults 128, 64.
    """

    CHANNEL_DEPTHS: Tuple[int, ...] = (2, 4, 8, 16)

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

        # ── Complex 2D conv trunk ─────────────────────────────────────────
        # Input has 3 complex channels (one per spatial direction).
        in_chs = [3] + list(self.CHANNEL_DEPTHS[:-1])
        self.blocks = nn.ModuleList([
            _ComplexConvBlock(c_in, c_out)
            for c_in, c_out in zip(in_chs, self.CHANNEL_DEPTHS)
        ])
        self.dropout = nn.Dropout(p=0.5)

        # ── Flat dim: last depth × 2 (real∥imag) × spatial ───────────────
        f_out, t_out = _pooled_dims(n_freq, n_time, len(self.CHANNEL_DEPTHS))
        flat_dim = self.CHANNEL_DEPTHS[-1] * 2 * f_out * t_out

        # ── 3-layer real-valued MLP ───────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim,     mlp_hidden_1), nn.ELU(inplace=True),
            nn.Linear(mlp_hidden_1, mlp_hidden_2), nn.ELU(inplace=True),
            nn.Linear(mlp_hidden_2, 2),
        )
        _init_mlp(self.mlp)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape ``(B, 6, F, T)`` — real active/reactive features.

        Returns
        -------
        Tensor
            Shape ``(B, 2)`` — ``(azimuth, elevation)`` in radians.
        """
        # ── Encode 6 real channels → 3 complex channels ──────────────────
        # z_d = active_d + j * reactive_d   (mirrors the cross-spectrum)
        h = torch.complex(x[:, :3], x[:, 3:])   # (B, 3, F, T) cfloat

        # ── Conv trunk ───────────────────────────────────────────────────
        for block in self.blocks:
            h = block(h)                          # (B, C, F', T') cfloat

        # ── Decode complex → real: concatenate real and imaginary parts ───
        h = torch.cat([h.real, h.imag], dim=1)   # (B, 2C, F', T')
        h = self.dropout(h)
        h = h.flatten(start_dim=1)

        return self.mlp(h)                        # (B, 2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return separate ``(azimuth, elevation)`` tensors."""
        out = self(x)
        return out[:, 0], out[:, 1]

    def count_parameters(self) -> int:
        """Trainable parameters, counting each complex weight as 2 reals."""
        total = 0
        for p in self.parameters():
            n = p.numel()
            if p.is_complex():
                n *= 2
            total += n
        return total

    def __repr__(self) -> str:
        return super().__repr__() + f"\n  Trainable parameters: {self.count_parameters():,}"


# ============================================================================
# BicomplexSpatialNet
# ============================================================================

class BicomplexSpatialNet(nn.Module):
    """Bicomplex-valued unified CNN for 3D Source Localization.

    **Encoding**: The 6 real intensity channels are packed into 2
    bicomplex channels in standard form ``(B, C_in×4, F, T)``:

    +-----------+------------------------------------------------------------+
    | BC ch 0   | ``q₀ = Ia_x + Ir_x·i₁ + Ia_y·i₂ + Ir_y·i₁i₂``           |
    +-----------+------------------------------------------------------------+
    | BC ch 1   | ``q₁ = Ia_z + Ir_z·i₁ + 0·i₂   + 0·i₁i₂``               |
    +-----------+------------------------------------------------------------+

    Channel 0 encodes the **horizontal-plane** DOA by algebraically
    coupling active and reactive intensity across both the x and y axes
    simultaneously.  Channel 1 encodes the **vertical axis**.

    Under the idempotent decomposition
    ``z₁ = (a+d)+i(b+c)``,  ``z₂ = (a-d)+i(b-c)``
    channel 0 becomes:

    * ``z₁ = (Ia_x + Ir_y) + j(Ir_x + Ia_y)``
    * ``z₂ = (Ia_x - Ir_y) + j(Ir_x - Ia_y)``

    These cross-sums (``Ia_x + Ir_y``, ``Ir_x + Ia_y``) are enforced by
    the *algebra itself* — not learned from scratch — making the
    bicomplex representation a physics-informed inductive bias.

    **Trunk**: Four :class:`_BiComplexConvBlock` stages, bicomplex channel
    depths ``(2, 4, 8, 16)`` (real channel depths ×4).  Standard-format
    output is already real, so no decode step is needed before the MLP.

    Parameters
    ----------
    n_freq : int
        STFT frequency bins ``F``.  Default 257.
    n_time : int
        STFT frames per 0.5 s chunk ``T``.  Default 63.
    mlp_hidden_1, mlp_hidden_2 : int
        MLP hidden-layer widths.  Defaults 128, 64.
    """

    CHANNEL_DEPTHS: Tuple[int, ...] = (2, 4, 8, 16)

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

        # ── BiComplex 2D conv trunk ───────────────────────────────────────
        # Input: 2 bicomplex channels → (B, 2×4=8, F, T) standard format.
        in_chs = [2] + list(self.CHANNEL_DEPTHS[:-1])
        self.blocks = nn.ModuleList([
            _BiComplexConvBlock(c_in, c_out)
            for c_in, c_out in zip(in_chs, self.CHANNEL_DEPTHS)
        ])
        self.dropout = nn.Dropout(p=0.5)

        # ── Flat dim: last_depth × 4 (standard form) × spatial ───────────
        f_out, t_out = _pooled_dims(n_freq, n_time, len(self.CHANNEL_DEPTHS))
        flat_dim = self.CHANNEL_DEPTHS[-1] * 4 * f_out * t_out

        # ── 3-layer real-valued MLP ───────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim,     mlp_hidden_1), nn.ELU(inplace=True),
            nn.Linear(mlp_hidden_1, mlp_hidden_2), nn.ELU(inplace=True),
            nn.Linear(mlp_hidden_2, 2),
        )
        _init_mlp(self.mlp)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape ``(B, 6, F, T)`` — real active/reactive features.

        Returns
        -------
        Tensor
            Shape ``(B, 2)`` — ``(azimuth, elevation)`` in radians.
        """
        # ── Encode 6 real channels → 2 bicomplex channels (standard form) ─
        #
        # Standard layout for C_in=2:  [a₀, a₁, b₀, b₁, c₀, c₁, d₀, d₁]
        #   a group: [Ia_x, Ia_z]    (Re(i₁=0, i₂=0) component)
        #   b group: [Ir_x, Ir_z]    (Im under i₁)
        #   c group: [Ia_y,  0  ]    (Im under i₂)
        #   d group: [Ir_y,  0  ]    (Im under i₁i₂)
        #
        Ia_x, Ia_y, Ia_z = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        Ir_x, Ir_y, Ir_z = x[:, 3:4], x[:, 4:5], x[:, 5:6]
        zeros = torch.zeros_like(Ia_x)

        h = torch.cat([
            Ia_x,  Ia_z,           # a group
            Ir_x,  Ir_z,           # b group
            Ia_y,  zeros,          # c group
            Ir_y,  zeros,          # d group
        ], dim=1)                  # (B, 8, F, T)

        # ── Conv trunk ───────────────────────────────────────────────────
        for block in self.blocks:
            h = block(h)           # (B, C_out×4, F', T') real

        # Standard format is already real — no decode step needed.
        h = self.dropout(h)
        h = h.flatten(start_dim=1)

        return self.mlp(h)         # (B, 2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return separate ``(azimuth, elevation)`` tensors."""
        out = self(x)
        return out[:, 0], out[:, 1]

    def count_parameters(self) -> int:
        """Trainable parameters, counting each complex weight as 2 reals."""
        total = 0
        for p in self.parameters():
            n = p.numel()
            if p.is_complex():
                n *= 2
            total += n
        return total

    def __repr__(self) -> str:
        return super().__repr__() + f"\n  Trainable parameters: {self.count_parameters():,}"
