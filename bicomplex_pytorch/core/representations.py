"""
This module provides utilities for working with bicomplex numbers represented as tensors.

A bicomplex number is represented in the form:
    z = a + b i + c j + d i j
where i and j are imaginary units such that i² = j² = -1 and i j = j i.

The module includes functions for creating, converting, extracting components,
and performing transformations on bicomplex tensors.
"""
import torch
from typing import Tuple


def is_bicomplex(tensor: torch.Tensor) -> bool:
    """
    Check if a tensor is in bicomplex representation format.

    A tensor is considered bicomplex if:
    - It has at least one dimension
    - The last dimension has exactly 4 components (a, b, c, d)
    - It contains real-valued (floating point or integer) data

    Args:
        tensor (torch.Tensor): The tensor to check.

    Returns:
        bool: True if the tensor is in bicomplex format, False otherwise.
    """
    # Check if tensor has at least one dimension
    if tensor.ndim == 0:
        return False

    # Check if last dimension has 4 components
    if tensor.shape[-1] != 4:
        return False

    # Check if tensor contains real values (not complex)
    if torch.is_complex(tensor):
        return False

    # Check if tensor is numeric (floating point or integer)
    if not (torch.is_floating_point(tensor) or tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64,
                                                                torch.uint8]):
        return False

    return True


def is_idempotent(value) -> bool:
    """
    Determines if the provided value is idempotent.

    An idempotent value, when used as an input, yields the same result (output)
    when applied repeatedly.

    Parameters:
    value: The input value to be checked for idempotency.

    Returns:
    bool: True if the value is idempotent, otherwise False.
    """
    if not isinstance(value, (tuple, list)):
        return False

    if len(value) != 2:
        return False

    z1, z2 = value

    # Check if both elements are tensors
    if not isinstance(z1, torch.Tensor) or not isinstance(z2, torch.Tensor):
        return False

    # Check if both tensors are complex-valued
    if not torch.is_complex(z1) or not torch.is_complex(z2):
        return False

    # Check if both tensors have the same shape
    if z1.shape != z2.shape:
        return False

    return True


def real_to_bicomplex(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a real-valued tensor to a bicomplex tensor representation.

    A bicomplex number is represented as:
        z = a + b i + c j + d i j

    This function interprets a real number `a` as a bicomplex number:
        z = a + 0 i + 0 j + 0 i j

    Args:
        tensor (torch.Tensor): A real-valued tensor.

    Returns:
        torch.Tensor: A tensor of shape `(..., 4)` where the last axis represents
                      the bicomplex components [a, 0, 0, 0].

    Raises:
        ValueError: If the input tensor is not real-valued.
    """
    if not torch.is_floating_point(tensor) and not torch.is_integer(tensor):
        raise ValueError("Input tensor must be real-valued.")

    a = tensor
    b = torch.zeros_like(a)
    c = torch.zeros_like(a)
    d = torch.zeros_like(a)

    return torch.stack([a, b, c, d], dim=-1)


def complex_to_bicomplex(complex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a complex-valued tensor to a bicomplex tensor representation.

    A bicomplex number is represented as:
        z = a + b i + c j + d i j
    where i and j are distinct imaginary units with i j = j i.

    This function interprets a complex number `a + b i` as a bicomplex number
    with zero imaginary part in the `j` and `i j` components:
        z = a + b i + 0 j + 0 i j

    Args:
        complex_tensor (torch.Tensor): A complex-valued tensor.

    Returns:
        torch.Tensor: A tensor of shape `(..., 4)` where the last axis represents
                      the bicomplex components [a, b, c, d].

    Raises:
        ValueError: If the input tensor is not complex-valued.
    """
    if not torch.is_complex(complex_tensor):
        raise ValueError("Input tensor must be complex-valued.")

    a = complex_tensor.real
    b = complex_tensor.imag
    c = torch.zeros_like(a)
    d = torch.zeros_like(a)

    bicomplex_tensor = torch.stack([a, b, c, d], dim=-1)

    return bicomplex_tensor


def complex_j_to_bicomplex(complex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a complex-valued tensor to a bicomplex tensor representation.

    A bicomplex number is represented as:
        z = a + b i + c j + d i j
    where i and j are distinct imaginary units with i j = j i.

    This function interprets a complex number `a + b i` as a bicomplex number
    with zero imaginary part in the `j` and `i j` components:
        z = a + b i + 0 j + 0 i j

    Args:
        complex_tensor (torch.Tensor): A complex-valued tensor.

    Returns:
        torch.Tensor: A tensor of shape `(..., 4)` where the last axis represents
                      the bicomplex components [a, b, c, d].

    Raises:
        ValueError: If the input tensor is not complex-valued.
    """
    if not torch.is_complex(complex_tensor):
        raise ValueError("Input tensor must be complex-valued.")

    a = complex_tensor.real
    b = torch.zeros_like(a)
    c = complex_tensor.imag
    d = torch.zeros_like(a)

    bicomplex_tensor = torch.stack([a, b, c, d], dim=-1)

    return bicomplex_tensor


def hyperbolic_to_bicomplex(complex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor in hyperbolic space to its equivalent in bicomplex space.

    This function takes a tensor representing complex numbers in the hyperbolic
    space and converts it to a tensor representing bicomplex numbers. It is
    assumed that the tensor uses the real and imaginary parts of the hyperbolic
    numbers along its last dimension. The operation is performed element-wise.

    Args:
        complex_tensor: A 2D or higher-dimensional tensor of float type. Each
        complex number to be converted is represented by its real and imaginary
        components along the last dimension.

    Returns:
        A tensor of the same shape as the input tensor, where each complex
        number has been converted to its bicomplex equivalent.

    Raises:
        ValueError: If the input tensor is not complex-valued.
    """
    if not torch.is_complex(complex_tensor):
        raise ValueError("Input tensor must be complex-valued.")

    a = complex_tensor.real
    b = torch.zeros_like(a)
    c = torch.zeros_like(a)
    d = complex_tensor.imag

    bicomplex_tensor = torch.stack([a, b, c, d], dim=-1)

    return bicomplex_tensor


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


def bicomplex_to_complex(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extracts the complex part (a + bi) from a bicomplex tensor.

    Converts z = a + bi + cj + dij to a + bi (discards j and ij components).

    Args:
        bicomplex_tensor (torch.Tensor): Tensor of shape (..., 4) with [a, b, c, d].

    Returns:
        torch.Tensor: Complex tensor representing a + bi.

    Raises:
        ValueError: If input is not a valid bicomplex tensor.
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor with shape (..., 4).")

    a = bicomplex_tensor[..., 0]
    b = bicomplex_tensor[..., 1]

    return torch.complex(a, b)


def bicomplex_to_real(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extracts the real part from a bicomplex tensor.

    Args:
        bicomplex_tensor (torch.Tensor): Tensor of shape (..., 4) with [a, b, c, d].

    Returns:
        torch.Tensor: Real tensor containing only the 'a' component.

    Raises:
        ValueError: If input is not a valid bicomplex tensor.
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor with shape (..., 4).")

    return bicomplex_tensor[..., 0]


def get_components(bicomplex_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract all four components of a bicomplex tensor.

    Args:
        bicomplex_tensor (torch.Tensor): Tensor of shape (..., 4).

    Returns:
        Tuple of (a, b, c, d) representing the bicomplex components.

    Raises:
        ValueError: If input is not a valid bicomplex tensor.
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor with shape (..., 4).")

    a = bicomplex_tensor[..., 0]
    b = bicomplex_tensor[..., 1]
    c = bicomplex_tensor[..., 2]
    d = bicomplex_tensor[..., 3]

    return a, b, c, d


def create_bicomplex(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """
    Create a bicomplex tensor from four component tensors.

    Constructs z = a + bi + cj + dij.

    Args:
        a, b, c, d: Real tensors of the same shape representing components.

    Returns:
        torch.Tensor: Bicomplex tensor of shape (..., 4).

    Raises:
        ValueError: If components don't have matching shapes or are complex.
    """
    if not (a.shape == b.shape == c.shape == d.shape):
        raise ValueError("All components must have the same shape.")

    for component in [a, b, c, d]:
        if torch.is_complex(component):
            raise ValueError("Components must be real-valued tensors.")

    return torch.stack([a, b, c, d], dim=-1)


def make_bicomplex(a: torch.Tensor, b: torch.Tensor = None,
                   c: torch.Tensor = None, d: torch.Tensor = None) -> torch.Tensor:
    """
    Construct a bicomplex tensor from its components.

    Creates a bicomplex number z = a + bi + cj + dij from individual components.
    Missing components default to zero.

    Args:
        a: Real part
        b: Coefficient of i (default: 0)
        c: Coefficient of j (default: 0)
        d: Coefficient of ij (default: 0)

    Returns:
        Bicomplex tensor of shape (..., 4)
    """
    if b is None:
        b = torch.zeros_like(a)
    if c is None:
        c = torch.zeros_like(a)
    if d is None:
        d = torch.zeros_like(a)

    # Ensure all components have the same shape
    if not (a.shape == b.shape == c.shape == d.shape):
        raise ValueError("All components must have the same shape")

    return torch.stack([a, b, c, d], dim=-1)