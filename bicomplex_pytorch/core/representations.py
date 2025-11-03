"""
Transformations between bicomplex representations.

This module implements the conversion between standard bicomplex
representation and the idempotent representation, which allows
treating bicomplex numbers as pairs of independent complex numbers.
"""
import torch
from typing import Tuple


def to_idempotent(bicomplex_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert bicomplex tensor to idempotent representation.

    Transforms a bicomplex number z = a + bi + cj + dij into
    the idempotent form (z₁, z₂) where z₁, z₂ are complex numbers.

    Args:
        bicomplex_tensor: Tensor of shape (..., 4) representing bicomplex
                         numbers with components [a, b, c, d]

    Returns:
        Tuple of two complex tensors (z1, z2) representing the
        idempotent components

    """
    a, b, c, d = bicomplex_tensor[..., 0], bicomplex_tensor[..., 1], \
        bicomplex_tensor[..., 2], bicomplex_tensor[..., 3]

    z1_real = a + d
    z1_imag = b + c
    z2_real = a - d
    z2_imag = b - c

    z1 = torch.complex(z1_real, z1_imag)
    z2 = torch.complex(z2_real, z2_imag)

    return z1, z2


def from_idempotent(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Convert from idempotent representation back to bicomplex.

    Transforms idempotent components (z₁, z₂) back to standard
    bicomplex form a + bi + cj + dij.

    Args:
        z1: Complex tensor representing first idempotent component
        z2: Complex tensor representing second idempotent component

    Returns:
        Tensor of shape (..., 4) with bicomplex components [a, b, c, d]

    """
    z1_real, z1_imag = z1.real, z1.imag
    z2_real, z2_imag = z2.real, z2.imag

    a = (z1_real + z2_real) / 2
    b = (z1_imag + z2_imag) / 2
    c = (z1_imag - z2_imag) / 2
    d = (z1_real - z2_real) / 2

    return torch.stack([a, b, c, d], dim=-1)


def to_cartesian(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Convert idempotent representation to Cartesian form.

    Alias for from_idempotent for clearer semantics in some contexts.

    Args:
        z1: First idempotent component
        z2: Second idempotent component

    Returns:
        Cartesian bicomplex tensor (a, b, c, d)
    """
    return from_idempotent(z1, z2)