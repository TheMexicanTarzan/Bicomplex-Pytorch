
"""
Utility functions for bicomplex neural networks.

Includes initialization, conversion, validation, and helper functions.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union, Literal

from ..core.representations import (
    is_idempotent,
    is_bicomplex,
    to_idempotent,
    from_idempotent,
)
from ..core.arithmetic import (
    modulus,
    modulus_squared,
    get_real_part,
    get_i_part,
    get_j_part,
    get_ij_part,
)


# ============================================================================
# Tensor Creation and Initialization
# ============================================================================

def zeros_idempotent(
        *shape,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a bicomplex tensor filled with zeros in idempotent form.

    Args:
        *shape: Shape of each component tensor
        dtype: Data type (should be complex)
        device: Device to create tensor on
        requires_grad: Whether to track gradients

    Returns:
        Tuple of zero tensors (e1, e2)
    """
    if dtype is None:
        dtype = torch.complex64

    e1 = torch.zeros(*shape, dtype=dtype, device=device, requires_grad=requires_grad)
    e2 = torch.zeros(*shape, dtype=dtype, device=device, requires_grad=requires_grad)

    return (e1, e2)


def ones_idempotent(
        *shape,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a bicomplex tensor filled with ones in idempotent form.

    For bicomplex 1 = (1, 1) in idempotent form.
    """
    if dtype is None:
        dtype = torch.complex64

    e1 = torch.ones(*shape, dtype=dtype, device=device, requires_grad=requires_grad)
    e2 = torch.ones(*shape, dtype=dtype, device=device, requires_grad=requires_grad)

    return (e1, e2)


def randn_idempotent(
        *shape,
        mean: float = 0.0,
        std: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a bicomplex tensor with normally distributed random values.

    Args:
        *shape: Shape of each component tensor
        mean: Mean of the normal distribution
        std: Standard deviation
        dtype: Data type (should be complex)
        device: Device to create tensor on
        requires_grad: Whether to track gradients

    Returns:
        Tuple of random tensors (e1, e2) with complex Gaussian noise
    """
    if dtype is None:
        dtype = torch.complex64

    # For complex tensors, create real and imaginary parts separately
    e1_real = torch.randn(*shape, device=device) * std + mean
    e1_imag = torch.randn(*shape, device=device) * std + mean
    e1 = torch.complex(e1_real, e1_imag)

    e2_real = torch.randn(*shape, device=device) * std + mean
    e2_imag = torch.randn(*shape, device=device) * std + mean
    e2 = torch.complex(e2_real, e2_imag)

    if requires_grad:
        e1.requires_grad_(True)
        e2.requires_grad_(True)

    return (e1, e2)


def rand_idempotent(
        *shape,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a bicomplex tensor with uniform random values in [0, 1).
    """
    if dtype is None:
        dtype = torch.complex64

    e1_real = torch.rand(*shape, device=device)
    e1_imag = torch.rand(*shape, device=device)
    e1 = torch.complex(e1_real, e1_imag)

    e2_real = torch.rand(*shape, device=device)
    e2_imag = torch.rand(*shape, device=device)
    e2 = torch.complex(e2_real, e2_imag)

    if requires_grad:
        e1.requires_grad_(True)
        e2.requires_grad_(True)

    return (e1, e2)


def randn_standard(
        *shape,
        mean: float = 0.0,
        std: float = 1.0,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> torch.Tensor:
    """
    Create a bicomplex tensor in standard form with random values.

    Returns:
        Tensor of shape (*shape, 4) with components [a, b, c, d]
    """
    tensor = torch.randn(*shape, 4, device=device) * std + mean

    if requires_grad:
        tensor.requires_grad_(True)

    return tensor


# ============================================================================
# Weight Initialization Methods
# ============================================================================

def xavier_uniform_idempotent(
        tensor: tuple[torch.Tensor, torch.Tensor],
        gain: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize bicomplex tensor using Xavier uniform initialization.

    Adapts PyTorch's xavier_uniform_ for bicomplex numbers.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Calculate fan_in and fan_out from tensor shape
    dimensions = tensor[0].dim()
    if dimensions < 2:
        raise ValueError("Xavier initialization requires at least 2D tensors")

    fan_in = tensor[0].shape[1]
    fan_out = tensor[0].shape[0]

    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    bound = np.sqrt(3.0) * std

    with torch.no_grad():
        # Initialize real and imaginary parts separately for each component
        tensor[0].real.uniform_(-bound, bound)
        tensor[0].imag.uniform_(-bound, bound)
        tensor[1].real.uniform_(-bound, bound)
        tensor[1].imag.uniform_(-bound, bound)

    return tensor


def xavier_normal_idempotent(
        tensor: tuple[torch.Tensor, torch.Tensor],
        gain: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize bicomplex tensor using Xavier normal initialization.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    dimensions = tensor[0].dim()
    if dimensions < 2:
        raise ValueError("Xavier initialization requires at least 2D tensors")

    fan_in = tensor[0].shape[1]
    fan_out = tensor[0].shape[0]

    std = gain * np.sqrt(2.0 / (fan_in + fan_out))

    with torch.no_grad():
        tensor[0].real.normal_(0, std)
        tensor[0].imag.normal_(0, std)
        tensor[1].real.normal_(0, std)
        tensor[1].imag.normal_(0, std)

    return tensor


def kaiming_uniform_idempotent(
        tensor: tuple[torch.Tensor, torch.Tensor],
        a: float = 0,
        mode: Literal['fan_in', 'fan_out'] = 'fan_in',
        nonlinearity: str = 'leaky_relu'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize bicomplex tensor using Kaiming uniform initialization.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    dimensions = tensor[0].dim()
    if dimensions < 2:
        raise ValueError("Kaiming initialization requires at least 2D tensors")

    fan_in = tensor[0].shape[1]
    fan_out = tensor[0].shape[0]

    fan = fan_in if mode == 'fan_in' else fan_out
    gain = torch.nn.init.calculate_gain(nonlinearity, a)

    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std

    with torch.no_grad():
        tensor[0].real.uniform_(-bound, bound)
        tensor[0].imag.uniform_(-bound, bound)
        tensor[1].real.uniform_(-bound, bound)
        tensor[1].imag.uniform_(-bound, bound)

    return tensor


def kaiming_normal_idempotent(
        tensor: tuple[torch.Tensor, torch.Tensor],
        a: float = 0,
        mode: Literal['fan_in', 'fan_out'] = 'fan_in',
        nonlinearity: str = 'leaky_relu'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize bicomplex tensor using Kaiming normal initialization.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    dimensions = tensor[0].dim()
    if dimensions < 2:
        raise ValueError("Kaiming initialization requires at least 2D tensors")

    fan_in = tensor[0].shape[1]
    fan_out = tensor[0].shape[0]

    fan = fan_in if mode == 'fan_in' else fan_out
    gain = torch.nn.init.calculate_gain(nonlinearity, a)

    std = gain / np.sqrt(fan)

    with torch.no_grad():
        tensor[0].real.normal_(0, std)
        tensor[0].imag.normal_(0, std)
        tensor[1].real.normal_(0, std)
        tensor[1].imag.normal_(0, std)

    return tensor


def uniform_idempotent(
        tensor: tuple[torch.Tensor, torch.Tensor],
        a: float = 0.0,
        b: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fill bicomplex tensor with uniform random values in [a, b).
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    with torch.no_grad():
        tensor[0].real.uniform_(a, b)
        tensor[0].imag.uniform_(a, b)
        tensor[1].real.uniform_(a, b)
        tensor[1].imag.uniform_(a, b)

    return tensor


def normal_idempotent(
        tensor: tuple[torch.Tensor, torch.Tensor],
        mean: float = 0.0,
        std: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fill bicomplex tensor with normal random values.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    with torch.no_grad():
        tensor[0].real.normal_(mean, std)
        tensor[0].imag.normal_(mean, std)
        tensor[1].real.normal_(mean, std)
        tensor[1].imag.normal_(mean, std)

    return tensor


# ============================================================================
# Conversion Utilities
# ============================================================================

def real_to_bicomplex_idempotent(
        real_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a real tensor to bicomplex form (pure real part).

    For real value r, bicomplex form is (r, r) in idempotent representation.
    """
    if real_tensor.is_complex():
        raise ValueError("Input tensor is already complex")

    # Convert to complex with zero imaginary part
    complex_tensor = real_tensor.to(torch.complex64)

    return (complex_tensor, complex_tensor)


def complex_to_bicomplex_idempotent(
        complex_tensor: torch.Tensor,
        axis: Literal['i', 'j'] = 'i'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a complex tensor to bicomplex form.

    Args:
        complex_tensor: Complex tensor
        axis: Which imaginary unit to use ('i' or 'j')

    Returns:
        Bicomplex tensor in idempotent form
    """
    if not complex_tensor.is_complex():
        raise ValueError("Input must be a complex tensor")

    # For a + bi, idempotent form is ((a+bi)/2 + (a-bi)/2, (a+bi)/2 - (a-bi)/2)
    # Which simplifies to (a + bi, a - bi)
    if axis == 'i':
        e1 = complex_tensor
        e2 = torch.conj(complex_tensor)
    else:  # axis == 'j'
        # For a + cj, idempotent form is (a + cj, a - cj)
        e1 = complex_tensor
        e2 = torch.conj(complex_tensor)

    return (e1, e2)


def tensor_to_bicomplex(
        tensor: torch.Tensor,
        representation: Literal['standard', 'idempotent'] = 'idempotent'
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert various tensor types to bicomplex representation.

    Args:
        tensor: Input tensor (real, complex, or shape (..., 4) for standard form)
        representation: Output representation ('standard' or 'idempotent')

    Returns:
        Bicomplex tensor in requested representation
    """
    # Check if already in standard bicomplex form
    if not tensor.is_complex() and tensor.shape[-1] == 4:
        if representation == 'standard':
            return tensor
        else:
            return to_idempotent(tensor)

    # Real tensor
    if not tensor.is_complex():
        idempotent = real_to_bicomplex_idempotent(tensor)
        if representation == 'idempotent':
            return idempotent
        else:
            return from_idempotent(idempotent)

    # Complex tensor
    idempotent = complex_to_bicomplex_idempotent(tensor)
    if representation == 'idempotent':
        return idempotent
    else:
        return from_idempotent(idempotent)


# ============================================================================
# Component Extraction and Analysis
# ============================================================================

def extract_components_standard(
        bicomplex_tensor: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Extract all components from a bicomplex tensor in standard form.

    Returns:
        Dictionary with keys 'real', 'i', 'j', 'ij'
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor")

    return {
        'real': get_real_part(bicomplex_tensor),
        'i': get_i_part(bicomplex_tensor),
        'j': get_j_part(bicomplex_tensor),
        'ij': get_ij_part(bicomplex_tensor)
    }


def extract_components_idempotent(
        bicomplex_tensor: tuple[torch.Tensor, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Extract components from idempotent form.

    Returns:
        Dictionary with keys 'e1_real', 'e1_imag', 'e2_real', 'e2_imag'
    """
    if not is_idempotent(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return {
        'e1_real': bicomplex_tensor[0].real,
        'e1_imag': bicomplex_tensor[0].imag,
        'e2_real': bicomplex_tensor[1].real,
        'e2_imag': bicomplex_tensor[1].imag
    }


def compute_statistics_idempotent(
        bicomplex_tensor: tuple[torch.Tensor, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Compute statistics for a bicomplex tensor.

    Returns:
        Dictionary with modulus statistics, component norms, etc.
    """
    if not is_idempotent(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    mod = modulus(bicomplex_tensor)
    mod_sq = modulus_squared(bicomplex_tensor)

    e1_norm = torch.abs(bicomplex_tensor[0])
    e2_norm = torch.abs(bicomplex_tensor[1])

    return {
        'modulus_mean': torch.mean(mod),
        'modulus_std': torch.std(mod),
        'modulus_max': torch.max(mod),
        'modulus_min': torch.min(mod),
        'modulus_squared_mean': torch.mean(mod_sq),
        'e1_norm_mean': torch.mean(e1_norm),
        'e2_norm_mean': torch.mean(e2_norm),
        'e1_real_mean': torch.mean(bicomplex_tensor[0].real),
        'e1_imag_mean': torch.mean(bicomplex_tensor[0].imag),
        'e2_real_mean': torch.mean(bicomplex_tensor[1].real),
        'e2_imag_mean': torch.mean(bicomplex_tensor[1].imag),
    }


# ============================================================================
# Gradient and Training Utilities
# ============================================================================

def clip_grad_norm_idempotent(
        tensor: tuple[torch.Tensor, torch.Tensor],
        max_norm: float,
        norm_type: float = 2.0
) -> torch.Tensor:
    """
    Clip gradient norm for bicomplex tensor.

    Args:
        tensor: Bicomplex tensor with gradients in idempotent form
        max_norm: Maximum norm value
        norm_type: Type of norm to use

    Returns:
        Total norm before clipping
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    parameters = [tensor[0], tensor[1]]
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)

    return total_norm


def zero_grad_idempotent(
        tensor: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """
    Zero out gradients for bicomplex tensor.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if tensor[0].grad is not None:
        tensor[0].grad.zero_()
    if tensor[1].grad is not None:
        tensor[1].grad.zero_()


def get_device(
        tensor: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
) -> torch.device:
    """
    Get device of bicomplex tensor (works for both representations).
    """
    if isinstance(tensor, tuple):
        return tensor[0].device
    else:
        return tensor.device


def to_device(
        tensor: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        device: torch.device
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Move bicomplex tensor to device.
    """
    if isinstance(tensor, tuple):
        return (tensor[0].to(device), tensor[1].to(device))
    else:
        return tensor.to(device)


# ============================================================================
# Debugging and Validation
# ============================================================================

def validate_bicomplex_tensor(
        tensor: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        name: str = "tensor"
) -> bool:
    """
    Validate that tensor is a proper bicomplex tensor and check for NaN/Inf.

    Args:
        tensor: Tensor to validate
        name: Name for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If tensor is invalid
    """
    if isinstance(tensor, tuple):
        if not is_idempotent(tensor):
            raise ValueError(f"{name} is not a valid bicomplex tensor in idempotent form")

        if torch.isnan(tensor[0]).any() or torch.isnan(tensor[1]).any():
            raise ValueError(f"{name} contains NaN values")

        if torch.isinf(tensor[0]).any() or torch.isinf(tensor[1]).any():
            raise ValueError(f"{name} contains Inf values")
    else:
        if not is_bicomplex(tensor):
            raise ValueError(f"{name} is not a valid bicomplex tensor in standard form")

        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")

        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains Inf values")

    return True


def print_bicomplex_info(
        tensor: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        name: str = "tensor"
) -> None:
    """
    Print detailed information about a bicomplex tensor.
    """
    print(f"\n{'= ' *60}")
    print(f"Bicomplex Tensor: {name}")
    print(f"{'= ' *60}")

    if isinstance(tensor, tuple):
        print(f"Representation: Idempotent")
        print(f"Shape (e1): {tensor[0].shape}")
        print(f"Shape (e2): {tensor[1].shape}")
        print(f"Dtype: {tensor[0].dtype}")
        print(f"Device: {tensor[0].device}")
        print(f"Requires grad: {tensor[0].requires_grad}")

        stats = compute_statistics_idempotent(tensor)
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value.item():.6f}")
    else:
        print(f"Representation: Standard")
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        print(f"Requires grad: {tensor.requires_grad}")

        components = extract_components_standard(tensor)
        print(f"\nComponent statistics:")
        for key, comp in components.items():
            print(f"  {key}: mean={comp.mean().item():.6f}, std={comp.std().item():.6f}")

    print(f"{'= ' *60}\n")