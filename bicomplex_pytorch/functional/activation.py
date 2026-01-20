"""
Functional activation functions for bicomplex neural networks.

All functions operate on bicomplex tensors in idempotent form (e1, e2).

================================================================================
BICOMPLEX DIFFERENTIABILITY AND ACTIVATION FUNCTIONS
================================================================================

THEORETICAL BACKGROUND:
-----------------------
A function f: BC → BC is BICOMPLEX-HOLOMORPHIC if it satisfies the generalized
Cauchy-Riemann conditions. In the idempotent representation:

    f(z) = f(z₁e₁ + z₂e₂) = f₁(z₁)e₁ + f₂(z₂)e₂

where f₁, f₂: C → C must each be COMPLEX-HOLOMORPHIC (analytic).

This means bicomplex-holomorphic functions decompose into two independent
complex-holomorphic functions on the idempotent components.

ACTIVATION FUNCTION CATEGORIES:
-------------------------------

1. BICOMPLEX-HOLOMORPHIC ACTIVATIONS (preserve algebraic structure):
   - bicomplex_exp_activation: f(z) = exp(z) - holomorphic
   - bicomplex_polynomial_activation: f(z) = Σaₖzᵏ - holomorphic
   - bicomplex_rational_activation: f(z) = p(z)/q(z) - holomorphic where q(z)≠0
   - bicomplex_sin_activation, bicomplex_cos_activation - holomorphic
   - bicomplex_sinh_activation, bicomplex_cosh_activation - holomorphic

2. SPLIT ACTIVATIONS (apply complex-holomorphic function to each component):
   - These ARE bicomplex-holomorphic by construction
   - split_sigmoid, split_tanh use the complex extensions

3. NON-HOLOMORPHIC ACTIVATIONS (practical but break algebraic structure):
   - bicomplex_relu, bicomplex_leaky_relu, bicomplex_elu, etc.
   - These apply real-valued functions to complex tensors
   - PyTorch's implementation: relu(a+bi) = max(0,a) + bi
   - VIOLATES Cauchy-Riemann: ∂u/∂x ≠ ∂v/∂y

   WHY USE NON-HOLOMORPHIC ACTIVATIONS?
   - Standard in complex/hypercomplex deep learning literature
   - Empirically effective for many tasks
   - Provide necessary nonlinearity for universal approximation
   - The "algebraic purity" is sacrificed for practical performance

GRADIENT COMPUTATION:
---------------------
PyTorch uses WIRTINGER CALCULUS for complex autograd:
    ∂L/∂z* (conjugate Wirtinger derivative)

This is appropriate for minimizing real-valued loss functions over complex
parameters, but these are NOT bicomplex-holomorphic gradients in the algebraic
sense. For optimization purposes, this is the correct approach.

RECOMMENDATIONS:
----------------
- For tasks requiring algebraic consistency: use holomorphic activations
- For general deep learning: non-holomorphic activations often work better
- Consider the task: signal processing may benefit from holomorphic properties;
  classification tasks often work fine with ReLU-family activations

================================================================================
"""

import torch
from typing import Optional, Literal, Callable

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

    Since e1 and e2 are complex tensors, we apply "split ReLU":
    ReLU(a + bi) = ReLU(a) + i*ReLU(b)

    Args:
        input: Bicomplex tensor in idempotent form
        inplace: If True, modifies input in-place

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    def complex_relu(z):
        """Apply ReLU to real and imaginary parts separately."""
        return torch.complex(torch.relu(z.real), torch.relu(z.imag))

    if inplace:
        # For inplace, we need to modify the underlying real/imag data
        input[0].real.relu_()
        input[0].imag.relu_()
        input[1].real.relu_()
        input[1].imag.relu_()
        return input

    return (complex_relu(input[0]), complex_relu(input[1]))


def bicomplex_leaky_relu(
    input: tuple[torch.Tensor, torch.Tensor],
    negative_slope: float = 0.01,
    inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Leaky ReLU activation component-wise.

    Since the idempotent components are complex tensors, we apply "split" leaky ReLU
    to the real and imaginary parts separately.

    Args:
        input: Bicomplex tensor in idempotent form
        negative_slope: Controls the slope for negative values
        inplace: If True, modifies input in-place

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    def complex_leaky_relu(z):
        """Apply Leaky ReLU to real and imaginary parts separately."""
        return torch.complex(
            torch.nn.functional.leaky_relu(z.real, negative_slope=negative_slope, inplace=inplace),
            torch.nn.functional.leaky_relu(z.imag, negative_slope=negative_slope, inplace=inplace)
        )

    return (complex_leaky_relu(input[0]), complex_leaky_relu(input[1]))


def bicomplex_elu(
    input: tuple[torch.Tensor, torch.Tensor],
    alpha: float = 1.0,
    inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies ELU activation component-wise.

    Since the idempotent components are complex tensors, we apply "split" ELU
    to the real and imaginary parts separately.

    Args:
        input: Bicomplex tensor in idempotent form
        alpha: Scale for negative values
        inplace: If True, modifies input in-place

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    def complex_elu(z):
        """Apply ELU to real and imaginary parts separately."""
        return torch.complex(
            torch.nn.functional.elu(z.real, alpha=alpha, inplace=inplace),
            torch.nn.functional.elu(z.imag, alpha=alpha, inplace=inplace)
        )

    return (complex_elu(input[0]), complex_elu(input[1]))


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


# =============================================================================
# BICOMPLEX-HOLOMORPHIC ACTIVATIONS
# =============================================================================
# These activations preserve bicomplex differentiability (satisfy generalized
# Cauchy-Riemann conditions). They are mathematically "pure" but may not always
# provide the best practical performance for all tasks.
# =============================================================================


def bicomplex_exp_activation(
    input: tuple[torch.Tensor, torch.Tensor],
    scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bicomplex exponential activation (HOLOMORPHIC).

    f(z) = exp(scale * z)

    This is a bicomplex-holomorphic function: exp is entire (holomorphic everywhere)
    on C, so applying it component-wise in idempotent form gives a bicomplex-
    holomorphic function.

    Args:
        input: Bicomplex tensor in idempotent form
        scale: Scaling factor applied before exponential (controls sensitivity)

    Returns:
        Activated bicomplex tensor

    Note:
        May produce very large values for large inputs. Consider using
        bicomplex_bounded_exp_activation for stability.
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if scale != 1.0:
        input = scalar_multiply_idempotent(scale, input)

    return exp_idempotent(input)


def bicomplex_bounded_exp_activation(
    input: tuple[torch.Tensor, torch.Tensor],
    scale: float = 1.0,
    max_magnitude: float = 10.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bounded bicomplex exponential activation (HOLOMORPHIC with clipping).

    Applies exp(scale * z) but clips input magnitude to prevent overflow.

    Args:
        input: Bicomplex tensor in idempotent form
        scale: Scaling factor applied before exponential
        max_magnitude: Maximum allowed magnitude for input components

    Returns:
        Activated bicomplex tensor with bounded output
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Clip the real parts to prevent overflow (imaginary parts affect phase only)
    e1_clipped = torch.complex(
        torch.clamp(input[0].real, min=-max_magnitude, max=max_magnitude),
        input[0].imag
    )
    e2_clipped = torch.complex(
        torch.clamp(input[1].real, min=-max_magnitude, max=max_magnitude),
        input[1].imag
    )

    clipped = (e1_clipped, e2_clipped)

    if scale != 1.0:
        clipped = scalar_multiply_idempotent(scale, clipped)

    return exp_idempotent(clipped)


def bicomplex_sin_activation(
    input: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bicomplex sine activation (HOLOMORPHIC).

    f(z) = sin(z) = (exp(iz) - exp(-iz)) / (2i)

    Sine is entire (holomorphic everywhere) on C, making this bicomplex-holomorphic.
    Bounded output for real inputs, but can be unbounded for complex inputs.

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Complex sine is implemented in PyTorch
    return (torch.sin(input[0]), torch.sin(input[1]))


def bicomplex_cos_activation(
    input: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bicomplex cosine activation (HOLOMORPHIC).

    f(z) = cos(z) = (exp(iz) + exp(-iz)) / 2

    Cosine is entire (holomorphic everywhere) on C, making this bicomplex-holomorphic.

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.cos(input[0]), torch.cos(input[1]))


def bicomplex_sinh_activation(
    input: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bicomplex hyperbolic sine activation (HOLOMORPHIC).

    f(z) = sinh(z) = (exp(z) - exp(-z)) / 2

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.sinh(input[0]), torch.sinh(input[1]))


def bicomplex_cosh_activation(
    input: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bicomplex hyperbolic cosine activation (HOLOMORPHIC).

    f(z) = cosh(z) = (exp(z) + exp(-z)) / 2

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.cosh(input[0]), torch.cosh(input[1]))


def bicomplex_tanh_holomorphic(
    input: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bicomplex hyperbolic tangent activation (HOLOMORPHIC).

    f(z) = tanh(z) = sinh(z) / cosh(z)

    This is holomorphic except at poles (where cosh(z) = 0).
    PyTorch's complex tanh handles this correctly.

    Note:
        This differs from bicomplex_tanh which may use non-holomorphic
        implementation depending on PyTorch version.

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.tanh(input[0]), torch.tanh(input[1]))


def bicomplex_polynomial_activation(
    input: tuple[torch.Tensor, torch.Tensor],
    coefficients: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bicomplex polynomial activation (HOLOMORPHIC).

    f(z) = a₀ + a₁z + a₂z² + ... + aₙzⁿ

    Polynomials are entire functions, making this bicomplex-holomorphic.
    The coefficients should be real for typical use cases.

    Args:
        input: Bicomplex tensor in idempotent form
        coefficients: List of polynomial coefficients [a₀, a₁, a₂, ...]
                      where aₖ is the coefficient of zᵏ

    Returns:
        Activated bicomplex tensor

    Example:
        >>> # f(z) = 1 + z + 0.5*z² (approximation of exp(z))
        >>> coeffs = [1.0, 1.0, 0.5]
        >>> output = bicomplex_polynomial_activation(input, coeffs)
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if len(coefficients) == 0:
        raise ValueError("At least one coefficient must be provided")

    # Start with constant term
    result = (
        torch.full_like(input[0], coefficients[0]),
        torch.full_like(input[1], coefficients[0])
    )

    # Add higher order terms using Horner's method for efficiency
    if len(coefficients) > 1:
        # Horner's method: a₀ + z(a₁ + z(a₂ + ...))
        result_e1 = torch.full_like(input[0], coefficients[-1])
        result_e2 = torch.full_like(input[1], coefficients[-1])

        for coeff in reversed(coefficients[:-1]):
            result_e1 = result_e1 * input[0] + coeff
            result_e2 = result_e2 * input[1] + coeff

        result = (result_e1, result_e2)

    return result


def bicomplex_learnable_polynomial_activation(
    input: tuple[torch.Tensor, torch.Tensor],
    coefficients: tuple[torch.Tensor, torch.Tensor],
    degree: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bicomplex polynomial with learnable bicomplex coefficients (HOLOMORPHIC).

    f(z) = c₀ + c₁z + c₂z² + ... + cₙzⁿ

    where cₖ are bicomplex numbers (also in idempotent form).

    Args:
        input: Bicomplex tensor in idempotent form, shape (..., features)
        coefficients: Bicomplex coefficients in idempotent form, shape (degree+1,)
        degree: Degree of polynomial (len(coefficients) - 1)

    Returns:
        Activated bicomplex tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(coefficients):
        raise ValueError("Coefficients must be in idempotent form")

    # Extract coefficient components
    c_e1, c_e2 = coefficients

    # Use Horner's method with bicomplex coefficients
    result_e1 = c_e1[degree].expand_as(input[0]).clone()
    result_e2 = c_e2[degree].expand_as(input[1]).clone()

    for k in range(degree - 1, -1, -1):
        result_e1 = result_e1 * input[0] + c_e1[k]
        result_e2 = result_e2 * input[1] + c_e2[k]

    return (result_e1, result_e2)


def bicomplex_rational_activation(
    input: tuple[torch.Tensor, torch.Tensor],
    numerator_coeffs: list[float],
    denominator_coeffs: list[float],
    eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bicomplex rational function activation (HOLOMORPHIC where defined).

    f(z) = p(z) / q(z)

    where p and q are polynomials. This is holomorphic except at poles
    where q(z) = 0.

    Args:
        input: Bicomplex tensor in idempotent form
        numerator_coeffs: Coefficients of numerator polynomial
        denominator_coeffs: Coefficients of denominator polynomial
        eps: Small value added to denominator for numerical stability

    Returns:
        Activated bicomplex tensor

    Example:
        >>> # f(z) = z / (1 + z²) - a smooth bounded activation
        >>> num = [0.0, 1.0]  # z
        >>> den = [1.0, 0.0, 1.0]  # 1 + z²
        >>> output = bicomplex_rational_activation(input, num, den)
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    numerator = bicomplex_polynomial_activation(input, numerator_coeffs)
    denominator = bicomplex_polynomial_activation(input, denominator_coeffs)

    # Add small value to prevent division by zero
    denom_e1 = denominator[0] + eps
    denom_e2 = denominator[1] + eps

    return (numerator[0] / denom_e1, numerator[1] / denom_e2)


def bicomplex_split_activation(
    input: tuple[torch.Tensor, torch.Tensor],
    activation_fn: Callable[[torch.Tensor], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply any complex activation function independently to each idempotent component.

    This construction is BICOMPLEX-HOLOMORPHIC if and only if activation_fn is
    complex-holomorphic (analytic).

    Args:
        input: Bicomplex tensor in idempotent form
        activation_fn: A function that operates on complex tensors.
                       Must be complex-holomorphic for result to be
                       bicomplex-holomorphic.

    Returns:
        Activated bicomplex tensor

    Example:
        >>> # Custom holomorphic activation
        >>> def my_activation(z):
        ...     return z / (1 + torch.abs(z))  # NOT holomorphic!
        >>> output = bicomplex_split_activation(input, my_activation)

    Warning:
        The user is responsible for ensuring activation_fn is holomorphic
        if bicomplex-holomorphic output is required.
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (activation_fn(input[0]), activation_fn(input[1]))