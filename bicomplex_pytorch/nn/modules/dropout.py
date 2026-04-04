"""
Complex and bicomplex dropout layers.

This module implements dropout for complex- and bicomplex-valued neural
networks. The key principle is that dropout masks are shared across all
components of a number so that entire complex or bicomplex units are
dropped together, preserving algebraic structure.

================================================================================
DROPOUT STRATEGIES
================================================================================

1. COMPLEX DROPOUT (ComplexDropout):
   - Generates a *real-valued* binary mask and applies it to both the real
     and imaginary parts of each complex element.
   - This ensures that the full complex number is either kept or zeroed,
     preventing broken phase information.

2. BICOMPLEX DROPOUT (BiComplexDropout):
   Standard form input:
   - Generates a mask of shape (..., features) and broadcasts it across
     all four bicomplex components [a, b, c, d].
   Idempotent form input:
   - Generates a single real mask and applies it identically to both
     idempotent components z1 and z2, so entire bicomplex units are
     kept or dropped as a whole.

================================================================================
"""
import torch
import torch.nn as nn
from typing import Literal, Union
from ...core.representations import to_idempotent, from_idempotent, is_idempotent, is_bicomplex


class ComplexDropout(nn.Module):
    """
    Dropout layer for complex-valued tensors.

    During training, randomly zeroes entire complex elements (both real and
    imaginary parts together) with probability ``p``. This preserves the
    phase structure of surviving elements.

    During evaluation, the layer is a no-op.

    Args:
        p: Probability of an element being zeroed. Default: 0.5
        inplace: If True, performs the operation in-place. Default: False

    Shape:
        - Input: ``(*, features)`` complex tensor
        - Output: same shape as input
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout probability must be between 0 and 1, got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x

        # Generate a real-valued mask on the magnitude, then broadcast to
        # both real and imaginary parts via multiplication.
        mask = torch.ones(x.shape, device=x.device, dtype=torch.float32)
        mask = torch.nn.functional.dropout(mask, self.p, True, self.inplace)

        if x.is_complex():
            mask = mask.to(x.real.dtype)
            return x * mask
        # Fallback: real tensor passed through (acts like standard dropout)
        return x * mask.to(x.dtype)

    def extra_repr(self) -> str:
        return f'p={self.p}, inplace={self.inplace}'


class BiComplexDropout(nn.Module):
    """
    Dropout layer for bicomplex-valued tensors.

    Generates a single dropout mask per bicomplex feature and applies it
    to all four components (a, b, c, d) or equivalently to both idempotent
    components (z1, z2). This ensures entire bicomplex units are kept or
    dropped as a whole, preserving the algebraic structure.

    Args:
        p: Probability of a bicomplex element being zeroed. Default: 0.5
        inplace: If True, performs the operation in-place. Default: False
        input_format: Expected input format ('standard' or 'idempotent').
                      Default: 'standard'
        output_format: Output format ('standard' or 'idempotent').
                       Default: 'standard'

    Shape:
        - Input (standard): ``(..., features, 4)``
        - Input (idempotent): tuple ``(z1, z2)`` of complex tensors with
          shape ``(..., features)``
        - Output: same format/shape as input
    """

    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
        input_format: Literal['standard', 'idempotent'] = 'standard',
        output_format: Literal['standard', 'idempotent'] = 'standard',
    ):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout probability must be between 0 and 1, got {p}")
        self.p = p
        self.inplace = inplace
        self.input_format = input_format
        self.output_format = output_format

    def forward(
        self,
        x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not self.training or self.p == 0.0:
            return x

        # ---------- idempotent input ----------
        if is_idempotent(x):
            z1, z2 = x
            # Mask shape matches the feature dimensions of z1/z2
            mask = torch.ones(z1.shape, device=z1.device, dtype=torch.float32)
            mask = torch.nn.functional.dropout(mask, self.p, True, self.inplace)
            mask = mask.to(z1.real.dtype)

            z1_out = z1 * mask
            z2_out = z2 * mask

            if self.output_format == 'standard':
                return from_idempotent(z1_out, z2_out)
            return (z1_out, z2_out)

        # ---------- standard (cartesian) input ----------
        if is_bicomplex(x):
            # x has shape (..., features, 4)
            # Mask over (..., features), broadcast across last dim
            mask_shape = x.shape[:-1]
            mask = torch.ones(mask_shape, device=x.device, dtype=x.dtype)
            mask = torch.nn.functional.dropout(mask, self.p, True, self.inplace)
            # Unsqueeze so it broadcasts over the 4 components
            x_out = x * mask.unsqueeze(-1)

            if self.output_format == 'idempotent':
                return to_idempotent(x_out)
            return x_out

        raise ValueError(
            f"Expected bicomplex input (shape (..., 4) or tuple of complex tensors), "
            f"got {type(x)}"
        )

    def extra_repr(self) -> str:
        return (f'p={self.p}, inplace={self.inplace}, '
                f'input_format={self.input_format}, '
                f'output_format={self.output_format}')
