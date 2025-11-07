"""
Functional activation functions for bicomplex neural networks.

All functions operate on bicomplex tensors in idempotent form (e1, e2).
"""

import torch
from typing import Optional, Literal

from ..core.arithmetic import (
    exp_idempotent,
    log_idempotent,
    modulus,
    modulus_squared,
    divide_idempotent,
    add_idempotent,
    scalar_multiply_idempotent,
    multiply_idempotent,
    sqrt_idempotent,
    power_idempotent,
)
from ..core.representations import is_idempotent


def bicomplex_relu(
    input: tuple[torch.Tensor, torch.Tensor],
    inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies ReLU activation component-wise to bicomplex tensor.

    For z = (e1, e2), ReLU(z) = (ReLU(e1), ReLU(e2))

    Args:
        input: Bicomplex tensor in idempotent form
        inplace: If True, modifies input in-place

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if inplace:
        input[0].relu_()
        input[1].relu_()
        return input

    return (torch.relu(input[0]), torch.relu(input[1]))


def bicomplex_leaky_relu(
    input: tuple[torch.Tensor, torch.Tensor],
    negative_slope: float = 0.01,
    inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Leaky ReLU activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        negative_slope: Controls the slope for negative values
        inplace: If True, modifies input in-place

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.leaky_relu(input[0], negative_slope=negative_slope, inplace=inplace),
        torch.nn.functional.leaky_relu(input[1], negative_slope=negative_slope, inplace=inplace)
    )


def bicomplex_elu(
    input: tuple[torch.Tensor, torch.Tensor],
    alpha: float = 1.0,
    inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies ELU activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        alpha: Scale for negative values
        inplace: If True, modifies input in-place

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.elu(input[0], alpha=alpha, inplace=inplace),
        torch.nn.functional.elu(input[1], alpha=alpha, inplace=inplace)
    )


def bicomplex_selu(
    input: tuple[torch.Tensor, torch.Tensor],
    inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies SELU activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        inplace: If True, modifies input in-place

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.selu(input[0], inplace=inplace),
        torch.nn.functional.selu(input[1], inplace=inplace)
    )


def bicomplex_gelu(
    input: tuple[torch.Tensor, torch.Tensor],
    approximate: Literal['none', 'tanh'] = 'none'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies GELU activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        approximate: Approximation method ('none' or 'tanh')

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.gelu(input[0], approximate=approximate),
        torch.nn.functional.gelu(input[1], approximate=approximate)
    )


def bicomplex_sigmoid(
    input: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies sigmoid activation component-wise.

    For z = (e1, e2), sigmoid(z) = (sigmoid(e1), sigmoid(e2))

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Activated bicomplex tensor with values in (0, 1) for each component
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.sigmoid(input[0]), torch.sigmoid(input[1]))


def bicomplex_tanh(
    input: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies tanh activation component-wise.

    For z = (e1, e2), tanh(z) = (tanh(e1), tanh(e2))

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Activated bicomplex tensor with values in (-1, 1) for each component
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.tanh(input[0]), torch.tanh(input[1]))


def bicomplex_softplus(
    input: tuple[torch.Tensor, torch.Tensor],
    beta: float = 1.0,
    threshold: float = 20.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Softplus activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        beta: Scale parameter
        threshold: Threshold above which to use linear approximation

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.softplus(input[0], beta=beta, threshold=threshold),
        torch.nn.functional.softplus(input[1], beta=beta, threshold=threshold)
    )


def bicomplex_softsign(
    input: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Softsign activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.softsign(input[0]),
        torch.nn.functional.softsign(input[1])
    )


def bicomplex_hardtanh(
    input: tuple[torch.Tensor, torch.Tensor],
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Hardtanh activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        min_val: Minimum value
        max_val: Maximum value
        inplace: If True, modifies input in-place

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.hardtanh(input[0], min_val=min_val, max_val=max_val, inplace=inplace),
        torch.nn.functional.hardtanh(input[1], min_val=min_val, max_val=max_val, inplace=inplace)
    )


def bicomplex_softmax(
    input: tuple[torch.Tensor, torch.Tensor],
    dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies softmax activation component-wise.

    Normalizes components independently along specified dimension.

    Args:
        input: Bicomplex tensor in idempotent form
        dim: Dimension along which to apply softmax

    Returns:
        Normalized bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.softmax(input[0], dim=dim),
        torch.nn.functional.softmax(input[1], dim=dim)
    )


def bicomplex_log_softmax(
    input: tuple[torch.Tensor, torch.Tensor],
    dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies log-softmax activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        dim: Dimension along which to apply log-softmax

    Returns:
        Log-normalized bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.log_softmax(input[0], dim=dim),
        torch.nn.functional.log_softmax(input[1], dim=dim)
    )


def bicomplex_swish(
    input: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Swish (SiLU) activation component-wise.

    Swish(x) = x * sigmoid(x)

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.silu(input[0]),
        torch.nn.functional.silu(input[1])
    )


def bicomplex_mish(
    input: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Mish activation component-wise.

    Mish(x) = x * tanh(softplus(x))

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.mish(input[0]),
        torch.nn.functional.mish(input[1])
    )


def bicomplex_prelu(
    input: tuple[torch.Tensor, torch.Tensor],
    weight: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Parametric ReLU activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        weight: Learnable negative slope parameter (bicomplex)

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.prelu(input[0], weight[0]),
        torch.nn.functional.prelu(input[1], weight[1])
    )


def bicomplex_rrelu(
    input: tuple[torch.Tensor, torch.Tensor],
    lower: float = 0.125,
    upper: float = 0.333333,
    training: bool = False,
    inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Randomized Leaky ReLU activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        lower: Lower bound of random slope
        upper: Upper bound of random slope
        training: Whether in training mode
        inplace: If True, modifies input in-place

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.rrelu(input[0], lower=lower, upper=upper, training=training, inplace=inplace),
        torch.nn.functional.rrelu(input[1], lower=lower, upper=upper, training=training, inplace=inplace)
    )


def bicomplex_threshold(
    input: tuple[torch.Tensor, torch.Tensor],
    threshold: float,
    value: float,
    inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies threshold activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        threshold: Threshold value
        value: Value to use when below threshold
        inplace: If True, modifies input in-place

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        torch.nn.functional.threshold(input[0], threshold=threshold, value=value, inplace=inplace),
        torch.nn.functional.threshold(input[1], threshold=threshold, value=value, inplace=inplace)
    )


# Bicomplex-specific activations using the algebra


def bicomplex_modulus_activation(
    input: tuple[torch.Tensor, torch.Tensor],
    keepdim: bool = False
) -> torch.Tensor:
    """
    Returns the total modulus of the bicomplex number.

    Reduces bicomplex to real: |z| = sqrt(|e1|² + |e2|²)

    Args:
        input: Bicomplex tensor in idempotent form
        keepdim: Whether to keep bicomplex form (broadcasts result to both components)

    Returns:
        Real tensor or bicomplex tensor with same modulus in both components
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    mod = modulus(input)

    if keepdim:
        return (mod, mod)
    return mod


def bicomplex_normalize(
    input: tuple[torch.Tensor, torch.Tensor],
    eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes bicomplex tensor to unit modulus.

    Returns z / |z| where |z| = sqrt(|e1|² + |e2|²)

    Args:
        input: Bicomplex tensor in idempotent form
        eps: Small constant to avoid division by zero

    Returns:
        Normalized bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    mod = modulus(input)
    mod = torch.clamp(mod, min=eps)

    return (input[0] / mod, input[1] / mod)


def bicomplex_phase_activation(
    input: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Phase-preserving activation: maintains direction, passes through magnitude.

    Returns (e1/|e1|) * |z| for both components, preserving relative phase.

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Phase-activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Get component magnitudes
    mag_e1 = torch.abs(input[0])
    mag_e2 = torch.abs(input[1])

    # Get total modulus
    total_mod = modulus(input)

    # Normalize each component by its magnitude, then scale by total modulus
    eps = 1e-12
    result_e1 = (input[0] / torch.clamp(mag_e1, min=eps)) * total_mod
    result_e2 = (input[1] / torch.clamp(mag_e2, min=eps)) * total_mod

    return (result_e1, result_e2)