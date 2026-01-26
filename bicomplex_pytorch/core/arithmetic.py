"""
This module provides utilities for arithmetic operations on bicomplex numbers
represented as tensors.

A bicomplex number is represented in the form:
    z = a + b i + c j + d i j
where i and j are imaginary units such that i² = j² = -1 and i j = j i.

The module includes functions for creating, converting, extracting components,
and performing transformations on bicomplex tensors.

IMPORTANT - Idempotent Convention Used in This Library:
=========================================================
This library uses a specific idempotent decomposition convention. The standard
bicomplex idempotent basis elements are:
    e₁ = (1 + ij)/2,  e₂ = (1 - ij)/2

For z = a + bi + cj + dij, the idempotent representation (z₁, z₂) is computed as:
    z₁ = (a + d) + (b + c)i  (coefficient of e₁)
    z₂ = (a - d) + (b - c)i  (coefficient of e₂)

Note: Some mathematical literature (e.g., Price 2001, Luna-Elizarrarás et al.)
uses a different sign convention for the imaginary parts. This implementation
is internally self-consistent and correctly implements bicomplex arithmetic,
but users should be aware when comparing with external references.

The key advantage of idempotent form is that multiplication becomes component-wise:
    (z₁, z₂) * (w₁, w₂) = (z₁*w₁, z₂*w₂)

This avoids numerical issues with zero divisors in standard form while preserving
the full algebraic structure of bicomplex numbers.

Zero Divisors (Null Cone):
==========================
Bicomplex numbers contain non-trivial zero divisors. The "null cone" is:
    NC = {z ∈ BC : z₁ = 0 or z₂ = 0}

Any z ∈ NC has no multiplicative inverse. Functions that require division
(inverse, divide, log, etc.) include optional zero divisor handling.
"""

import torch
from typing import Optional
import warnings

from .representations import is_bicomplex, is_idempotent, to_idempotent, from_idempotent


# =============================================================================
# Zero Divisor Detection and Handling
# =============================================================================

class ZeroDivisorError(Exception):
    """Exception raised when an operation encounters a zero divisor."""
    pass


def is_zero_divisor(
    tensor: tuple[torch.Tensor, torch.Tensor],
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Check if bicomplex numbers are zero divisors (on the null cone).

    A bicomplex number z = z₁e₁ + z₂e₂ is a zero divisor if and only if
    z₁ = 0 or z₂ = 0 (i.e., it lies on the null cone).

    Args:
        tensor: Bicomplex tensor in idempotent form (e1, e2)
        eps: Threshold below which a component is considered zero

    Returns:
        Boolean tensor of same shape as input components, True where
        the bicomplex number is a zero divisor

    Example:
        >>> e1 = torch.complex(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0]))
        >>> e2 = torch.complex(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0]))
        >>> z = (e1, e2)
        >>> is_zero_divisor(z)
        tensor([False,  True])  # Second element has e1=0
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1, e2 = tensor

    # Compute magnitudes of each component
    mag_e1 = torch.abs(e1)
    mag_e2 = torch.abs(e2)

    # A number is a zero divisor if either component is (near) zero
    return (mag_e1 < eps) | (mag_e2 < eps)


def has_zero_divisor(
    tensor: tuple[torch.Tensor, torch.Tensor],
    eps: float = 1e-10
) -> bool:
    """
    Check if any element in the tensor is a zero divisor.

    Args:
        tensor: Bicomplex tensor in idempotent form
        eps: Threshold for zero detection

    Returns:
        True if any element is a zero divisor, False otherwise
    """
    return torch.any(is_zero_divisor(tensor, eps)).item()


def regularize_zero_divisors(
    tensor: tuple[torch.Tensor, torch.Tensor],
    eps: float = 1e-10,
    regularization: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Regularize zero divisors by adding small values to near-zero components.

    This is useful for operations like division and logarithm that would
    otherwise fail on zero divisors. The regularization preserves the
    direction of the component while ensuring it's non-zero.

    Args:
        tensor: Bicomplex tensor in idempotent form
        eps: Threshold below which a component is considered zero
        regularization: Value to add to near-zero components

    Returns:
        Regularized bicomplex tensor in idempotent form

    Note:
        This modifies the mathematical value slightly. For strict algebraic
        correctness, use error handling instead of regularization.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1, e2 = tensor

    # Find near-zero components
    near_zero_e1 = torch.abs(e1) < eps
    near_zero_e2 = torch.abs(e2) < eps

    # Regularize by adding small complex value
    reg_value = torch.complex(
        torch.tensor(regularization, dtype=e1.real.dtype, device=e1.device),
        torch.tensor(0.0, dtype=e1.real.dtype, device=e1.device)
    )

    e1_reg = torch.where(near_zero_e1, e1 + reg_value, e1)
    e2_reg = torch.where(near_zero_e2, e2 + reg_value, e2)

    return (e1_reg, e2_reg)


def get_real_part(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract the real part (a) from a bicomplex tensor.

    Args:
        bicomplex_tensor: Tensor of shape (..., 4) with components [a, b, c, d]

    Returns:
        Tensor of shape (...) containing only the real part
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor with shape (..., 4)")
    return bicomplex_tensor[..., 0]


def get_i_part(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract the i coefficient (b) from a bicomplex tensor.

    Args:
        bicomplex_tensor: Tensor of shape (..., 4) with components [a, b, c, d]

    Returns:
        Tensor of shape (...) containing the i coefficient
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor with shape (..., 4)")
    return bicomplex_tensor[..., 1]


def get_j_part(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract the j coefficient (c) from a bicomplex tensor.

    Args:
        bicomplex_tensor: Tensor of shape (..., 4) with components [a, b, c, d]

    Returns:
        Tensor of shape (...) containing the j coefficient
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor with shape (..., 4)")
    return bicomplex_tensor[..., 2]


def get_ij_part(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract the ij coefficient (d) from a bicomplex tensor.

    Args:
        bicomplex_tensor: Tensor of shape (..., 4) with components [a, b, c, d]

    Returns:
        Tensor of shape (...) containing the ij coefficient
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor with shape (..., 4)")
    return bicomplex_tensor[..., 3]


def bicomplex_to_complex_i(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Project bicomplex tensor to complex using i as imaginary unit.

    Extracts z = a + bi from z = a + bi + cj + dij

    Args:
        bicomplex_tensor: Tensor of shape (..., 4)

    Returns:
        Complex tensor containing a + bi
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor")

    a = bicomplex_tensor[..., 0]
    b = bicomplex_tensor[..., 1]

    return torch.complex(a, b)


def bicomplex_to_complex_j(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Project bicomplex tensor to complex using j as imaginary unit.

    Extracts z = a + cj from z = a + bi + cj + dij

    Args:
        bicomplex_tensor: Tensor of shape (..., 4)

    Returns:
        Complex tensor containing a + cj
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor")

    a = bicomplex_tensor[..., 0]
    c = bicomplex_tensor[..., 2]

    return torch.complex(a, c)


def bicomplex_to_real(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract the real scalar part from a bicomplex tensor.

    Alias for get_real_part for consistency with other conversion functions.

    Args:
        bicomplex_tensor: Tensor of shape (..., 4)

    Returns:
        Real tensor of shape (...)
    """
    return get_real_part(bicomplex_tensor)

def bicomplex_to_hyperbolic(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Project bicomplex tensor to hyperbolic space.

    Extracts z = a + dij from z = a + bi + cj + dij

    Args:
        bicomplex_tensor: Tensor of shape (..., 4)

    Returns:
        Hyperbolic tensor containing a + dij
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor")

    a = bicomplex_tensor[..., 0]
    b = bicomplex_tensor[..., 3]

    return torch.complex(a, b)


def add_standard(*tensors: torch.Tensor) -> torch.Tensor:
    """
    Adds multiple bicomplex tensors.

    Args:
        *tensors: Variable number of bicomplex tensors, each of shape (..., 4)

    Returns:
        Bicomplex tensor of shape (..., 4) representing the sum of all inputs

    Raises:
        ValueError: If no tensors provided or if inputs are not valid bicomplex tensors
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided")

    if len(tensors) == 1:
        if not is_bicomplex(tensors[0]):
            raise ValueError("Input must be a bicomplex tensor")
        return tensors[0]

    # Validate all tensors
    for i, tensor in tensors:
        if not is_bicomplex(tensor):
            raise ValueError(f"Tensor {i} must be a valid bicomplex tensor")

    # Add all tensors
    stacked_tensors = torch.stack(tensors, dim=0)
    return torch.sum(stacked_tensors, dim=0)


def add_idempotent(*tensors: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adds multiple bicomplex tensors in idempotent form.

    In idempotent form, a bicomplex number is represented as (e1, e2) where
    e1 and e2 are the coefficients of the idempotent basis elements.
    Addition is component-wise: (e1, e2) + (e1', e2') = (e1 + e1', e2 + e2')

    Args:
        *tensors: Variable number of bicomplex tensors in idempotent form,
                 each as a tuple (e1, e2) where e1 and e2 have shape (...)

    Returns:
        Tuple (e1, e2) of tensors with shape (...) representing the sum in idempotent form

    Raises:
        ValueError: If no tensors provided or if inputs are not valid bicomplex tensors
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided")

    if len(tensors) == 1:
        if not is_idempotent(tensors[0]):
            raise ValueError("Input must be a bicomplex tensor in idempotent form")
        return tensors[0]

    # Validate all tensors
    for i, tensor in enumerate(tensors):
        if not is_idempotent(tensor):
            raise ValueError(f"Tensor {i} must be a valid bicomplex tensor in idempotent form")

    # Add all e1 components and all e2 components separately
    e1_tensors = [t[0] for t in tensors]
    e2_tensors = [t[1] for t in tensors]

    stacked_e1 = torch.stack(e1_tensors, dim=0)
    stacked_e2 = torch.stack(e2_tensors, dim=0)

    result_e1 = torch.sum(stacked_e1, dim=0)
    result_e2 = torch.sum(stacked_e2, dim=0)

    return (result_e1, result_e2)


def multiply_idempotent(*tensors: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multiplies multiple bicomplex tensors in idempotent form.

    In idempotent form, a bicomplex number is represented as (e1, e2) where
    e1 and e2 are the coefficients of the idempotent basis elements.
    Multiplication is component-wise: (e1, e2) * (e1', e2') = (e1 * e1', e2 * e2')

    This works because the idempotent elements satisfy:
    - e₁ * e₁ = e₁
    - e₂ * e₂ = e₂
    - e₁ * e₂ = 0

    Args:
        *tensors: Variable number of bicomplex tensors in idempotent form,
                 each as a tuple (e1, e2) where e1 and e2 have shape (...)

    Returns:
        Tuple (e1, e2) of tensors with shape (...) representing the product in idempotent form

    Raises:
        ValueError: If no tensors provided or if inputs are not valid bicomplex tensors
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided")

    if len(tensors) == 1:
        if not is_idempotent(tensors[0]):
            raise ValueError("Input must be a bicomplex tensor in idempotent form")
        return tensors[0]

    # Validate all tensors
    for i, tensor in enumerate(tensors):
        if not is_idempotent(tensor):
            raise ValueError(f"Tensor {i} must be a valid bicomplex tensor in idempotent form")

    # Multiply all e1 components and all e2 components separately
    result_e1 = tensors[0][0]
    result_e2 = tensors[0][1]

    for tensor in tensors[1:]:
        result_e1 = result_e1 * tensor[0]
        result_e2 = result_e2 * tensor[1]

    return (result_e1, result_e2)


def subtract_idempotent(*tensors: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Subtracts bicomplex tensors in idempotent form (first - second - third - ...).
    """
    if len(tensors) < 2:
        raise ValueError("At least two tensors must be provided for subtraction")

    for i, tensor in enumerate(tensors):
        if not is_idempotent(tensor):
            raise ValueError(f"Tensor {i} must be a valid bicomplex tensor in idempotent form")

    result_e1 = tensors[0][0]
    result_e2 = tensors[0][1]

    for tensor in tensors[1:]:
        result_e1 = result_e1 - tensor[0]
        result_e2 = result_e2 - tensor[1]

    return (result_e1, result_e2)


def inverse_idempotent(
    tensor: tuple[torch.Tensor, torch.Tensor],
    zero_divisor_handling: str = 'error',
    eps: float = 1e-10,
    regularization: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the multiplicative inverse of a bicomplex tensor in idempotent form.

    For z = (e1, e2), the inverse is z⁻¹ = (1/e1, 1/e2).

    IMPORTANT: Zero divisors (where e1=0 or e2=0) have no inverse.
    This function provides several strategies for handling them.

    Args:
        tensor: Bicomplex tensor in idempotent form (e1, e2)
        zero_divisor_handling: Strategy for handling zero divisors:
            - 'error': Raise ZeroDivisorError if any zero divisor is found
            - 'warn': Issue warning and regularize zero divisors
            - 'regularize': Silently regularize zero divisors
            - 'ignore': Proceed without checking (may produce inf/nan)
        eps: Threshold for detecting zero divisors
        regularization: Value added to regularize near-zero components

    Returns:
        Tuple (1/e1, 1/e2) representing z⁻¹ in idempotent form

    Raises:
        ZeroDivisorError: If zero_divisor_handling='error' and zero divisors found
        ValueError: If input is not in valid idempotent form

    Example:
        >>> z = (torch.complex(torch.tensor([2.0]), torch.tensor([1.0])),
        ...      torch.complex(torch.tensor([3.0]), torch.tensor([0.0])))
        >>> inv_z = inverse_idempotent(z)  # (1/(2+i), 1/(3+0i))
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Handle zero divisors according to specified strategy
    if zero_divisor_handling != 'ignore':
        if has_zero_divisor(tensor, eps):
            if zero_divisor_handling == 'error':
                raise ZeroDivisorError(
                    "Cannot compute inverse: tensor contains zero divisors "
                    "(elements on the null cone where e1=0 or e2=0). "
                    "Use zero_divisor_handling='regularize' to handle this."
                )
            elif zero_divisor_handling == 'warn':
                warnings.warn(
                    "Tensor contains zero divisors. Applying regularization.",
                    RuntimeWarning
                )
                tensor = regularize_zero_divisors(tensor, eps, regularization)
            elif zero_divisor_handling == 'regularize':
                tensor = regularize_zero_divisors(tensor, eps, regularization)

    e1, e2 = tensor
    return (1.0 / e1, 1.0 / e2)


def divide_idempotent(
    numerator: tuple[torch.Tensor, torch.Tensor],
    denominator: tuple[torch.Tensor, torch.Tensor],
    zero_divisor_handling: str = 'error',
    eps: float = 1e-10,
    regularization: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Divides two bicomplex tensors in idempotent form.

    Division is component-wise: (e1, e2) / (e1', e2') = (e1/e1', e2/e2')

    IMPORTANT: Division by a zero divisor (where denominator's e1'=0 or e2'=0)
    is undefined. This function provides several strategies for handling them.

    Args:
        numerator: Bicomplex tensor in idempotent form (dividend)
        denominator: Bicomplex tensor in idempotent form (divisor)
        zero_divisor_handling: Strategy for handling zero divisors in denominator:
            - 'error': Raise ZeroDivisorError if any zero divisor is found
            - 'warn': Issue warning and regularize zero divisors
            - 'regularize': Silently regularize zero divisors
            - 'ignore': Proceed without checking (may produce inf/nan)
        eps: Threshold for detecting zero divisors
        regularization: Value added to regularize near-zero components

    Returns:
        Tuple (e1/e1', e2/e2') representing the quotient in idempotent form

    Raises:
        ZeroDivisorError: If zero_divisor_handling='error' and denominator
                          contains zero divisors
        ValueError: If inputs are not in valid idempotent form

    Example:
        >>> num = (torch.complex(torch.tensor([4.0]), torch.tensor([2.0])),
        ...        torch.complex(torch.tensor([6.0]), torch.tensor([3.0])))
        >>> den = (torch.complex(torch.tensor([2.0]), torch.tensor([1.0])),
        ...        torch.complex(torch.tensor([3.0]), torch.tensor([1.0])))
        >>> divide_idempotent(num, den)  # ((4+2i)/(2+i), (6+3i)/(3+i))
    """
    if not is_idempotent(numerator):
        raise ValueError("Numerator must be a bicomplex tensor in idempotent form")
    if not is_idempotent(denominator):
        raise ValueError("Denominator must be a bicomplex tensor in idempotent form")

    # Handle zero divisors in denominator according to specified strategy
    if zero_divisor_handling != 'ignore':
        if has_zero_divisor(denominator, eps):
            if zero_divisor_handling == 'error':
                raise ZeroDivisorError(
                    "Cannot divide: denominator contains zero divisors "
                    "(elements on the null cone where e1=0 or e2=0). "
                    "Use zero_divisor_handling='regularize' to handle this."
                )
            elif zero_divisor_handling == 'warn':
                warnings.warn(
                    "Denominator contains zero divisors. Applying regularization.",
                    RuntimeWarning
                )
                denominator = regularize_zero_divisors(denominator, eps, regularization)
            elif zero_divisor_handling == 'regularize':
                denominator = regularize_zero_divisors(denominator, eps, regularization)

    return (numerator[0] / denominator[0], numerator[1] / denominator[1])


def conjugate_i_standard(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute i-conjugate: changes sign of i component.

    (a + bi + cj + dij)* = a - bi + cj - dij
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor")

    result = bicomplex_tensor.clone()
    result[..., 1] = -result[..., 1]  # negate i component
    result[..., 3] = -result[..., 3]  # negate ij component
    return result


def conjugate_j_standard(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute j-conjugate: changes sign of j component.

    (a + bi + cj + dij)† = a + bi - cj - dij
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor")

    result = bicomplex_tensor.clone()
    result[..., 2] = -result[..., 2]  # negate j component
    result[..., 3] = -result[..., 3]  # negate ij component
    return result


def conjugate_total_standard(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute total conjugate: changes sign of both i and j.

    (a + bi + cj + dij)‡ = a - bi - cj + dij
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor")

    result = bicomplex_tensor.clone()
    result[..., 1] = -result[..., 1]  # negate i component
    result[..., 2] = -result[..., 2]  # negate j component
    return result


def conjugate_idempotent(tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Conjugate in idempotent form: swaps e1 and e2.

    (e1, e2)* = (e2, e1)
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (tensor[1], tensor[0])


def norm_idempotent(tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the bicomplex norm in idempotent form.

    For z = (e1, e2), norm(z) = z * conjugate(z) = (e1*e2, e2*e1) = (e1*e2, e1*e2)
    Returns the product e1*e2 as both components.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    product = tensor[0] * tensor[1]
    return (product, product)


def modulus_squared(z: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Computes the squared Euclidean modulus: |e1|² + |e2|²

    This is the EUCLIDEAN NORM when viewing BC as R⁴, NOT the bicomplex D-modulus.

    Mathematical Note:
    ==================
    The standard bicomplex D-modulus is defined as |z|_D = √(|z₁|·|z₂|), while
    the hyperbolic modulus is |z|² = z·z̄ which can be negative or complex.

    This implementation uses the Euclidean/L² norm:
        |z|² = |z₁|² + |z₂|² = a² + b² + c² + d²

    This choice is motivated by:
    1. Always non-negative (suitable for loss functions)
    2. Compatible with standard deep learning optimization
    3. Smooth gradient everywhere

    For complex tensors e1, e2:
        |e1|² = Re(e1)² + Im(e1)²

    Args:
        z: Bicomplex tensor in idempotent form (e1, e2)

    Returns:
        Real tensor containing squared Euclidean modulus
    """
    if z[0].is_complex():
        e1_sq = z[0].real ** 2 + z[0].imag ** 2
        e2_sq = z[1].real ** 2 + z[1].imag ** 2
    else:
        e1_sq = z[0] ** 2
        e2_sq = z[1] ** 2

    return e1_sq + e2_sq


def modulus(z: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Computes the Euclidean modulus: √(|e1|² + |e2|²)

    This is the EUCLIDEAN NORM when viewing BC as R⁴, NOT the bicomplex D-modulus.
    See modulus_squared for detailed mathematical discussion.

    Args:
        z: Bicomplex tensor in idempotent form (e1, e2)

    Returns:
        Real tensor containing Euclidean modulus
    """
    return torch.sqrt(modulus_squared(z))


def d_modulus(z: tuple[torch.Tensor, torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    """
    Computes the bicomplex D-modulus: √(|e1|·|e2|)

    This is the standard bicomplex modulus from the literature, defined as:
        |z|_D = √(|z₁|·|z₂|)

    Properties:
    - |z|_D = 0 if and only if z is a zero divisor
    - |z·w|_D = |z|_D · |w|_D (multiplicative)
    - NOT a norm (doesn't satisfy triangle inequality)

    Args:
        z: Bicomplex tensor in idempotent form (e1, e2)
        eps: Small value for numerical stability

    Returns:
        Real tensor containing D-modulus

    Note:
        For neural network loss functions, the Euclidean modulus() is usually preferred.
    """
    mag_e1 = torch.abs(z[0])
    mag_e2 = torch.abs(z[1])
    return torch.sqrt(mag_e1 * mag_e2 + eps)

def squared_idempotent_norm(z: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Component-wise norms: (|e1|², |e2|²)"""
    return (torch.abs(z[0])**2, torch.abs(z[1])**2)

def scalar_multiply_idempotent(scalar: float,
                               tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multiplies a bicomplex tensor by a real scalar.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (scalar * tensor[0], scalar * tensor[1])


def negate_idempotent(tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Negates a bicomplex tensor in idempotent form.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (-tensor[0], -tensor[1])


def exp_idempotent(tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the exponential of a bicomplex tensor in idempotent form.

    For z = (e1, e2), exp(z) = (exp(e1), exp(e2))

    Args:
        tensor: Bicomplex tensor in idempotent form (e1, e2)

    Returns:
        Tuple (exp(e1), exp(e2)) representing exp(z) in idempotent form
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.exp(tensor[0]), torch.exp(tensor[1]))


def log_idempotent(
    tensor: tuple[torch.Tensor, torch.Tensor],
    zero_divisor_handling: str = 'error',
    eps: float = 1e-10,
    regularization: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the natural logarithm of a bicomplex tensor in idempotent form.

    For z = (e1, e2), log(z) = (log(e1), log(e2))

    IMPORTANT: The logarithm is undefined for zero divisors (where e1=0 or e2=0).
    This function provides several strategies for handling them.

    Args:
        tensor: Bicomplex tensor in idempotent form (e1, e2)
        zero_divisor_handling: Strategy for handling zero divisors:
            - 'error': Raise ZeroDivisorError if any zero divisor is found
            - 'warn': Issue warning and regularize zero divisors
            - 'regularize': Silently regularize zero divisors
            - 'ignore': Proceed without checking (may produce -inf/nan)
        eps: Threshold for detecting zero divisors
        regularization: Value added to regularize near-zero components

    Returns:
        Tuple (log(e1), log(e2)) representing log(z) in idempotent form

    Raises:
        ZeroDivisorError: If zero_divisor_handling='error' and zero divisors found

    Note:
        For complex inputs, computes the principal value of the complex logarithm.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Handle zero divisors according to specified strategy
    if zero_divisor_handling != 'ignore':
        if has_zero_divisor(tensor, eps):
            if zero_divisor_handling == 'error':
                raise ZeroDivisorError(
                    "Cannot compute logarithm: tensor contains zero divisors "
                    "(elements on the null cone where e1=0 or e2=0). "
                    "Use zero_divisor_handling='regularize' to handle this."
                )
            elif zero_divisor_handling == 'warn':
                warnings.warn(
                    "Tensor contains zero divisors. Applying regularization.",
                    RuntimeWarning
                )
                tensor = regularize_zero_divisors(tensor, eps, regularization)
            elif zero_divisor_handling == 'regularize':
                tensor = regularize_zero_divisors(tensor, eps, regularization)

    e1, e2 = tensor

    # Handle complex logarithm - torch.log works for both real and complex tensors
    # For complex inputs, it computes the principal value
    log_e1 = torch.log(e1)
    log_e2 = torch.log(e2)

    return (log_e1, log_e2)


def exp_standard(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the exponential of a bicomplex tensor in standard form.

    Uses the identity: exp(a + bi + cj + dij) can be computed by:
    1. Converting to idempotent form
    2. Applying component-wise exponential
    3. Converting back to standard form

    Args:
        bicomplex_tensor: Tensor of shape (..., 4) with components [a, b, c, d]

    Returns:
        Bicomplex tensor of shape (..., 4) representing exp(z)
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor")

    # Convert to idempotent form
    idempotent = to_idempotent(bicomplex_tensor)

    # Apply exponential component-wise
    exp_idempotent_result = exp_idempotent(idempotent)

    # Convert back to standard form
    return from_idempotent(exp_idempotent_result)


def log_standard(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the natural logarithm of a bicomplex tensor in standard form.

    Uses the identity by converting to idempotent form, applying log,
    and converting back.

    Args:
        bicomplex_tensor: Tensor of shape (..., 4) with components [a, b, c, d]

    Returns:
        Bicomplex tensor of shape (..., 4) representing log(z)
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor")

    # Convert to idempotent form
    idempotent = to_idempotent(bicomplex_tensor)

    # Apply logarithm component-wise
    log_idempotent_result = log_idempotent(idempotent)

    # Convert back to standard form
    return from_idempotent(log_idempotent_result)


def power_idempotent(base: tuple[torch.Tensor, torch.Tensor],
                     exponent: float | int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Raises a bicomplex tensor to a power in idempotent form.

    For z = (e1, e2), z^n = (e1^n, e2^n)

    Args:
        base: Bicomplex tensor in idempotent form (e1, e2)
        exponent: Real scalar, integer, or tensor

    Returns:
        Tuple (e1^n, e2^n) representing z^n in idempotent form
    """
    if not is_idempotent(base):
        raise ValueError("Base must be a bicomplex tensor in idempotent form")

    return (torch.pow(base[0], exponent), torch.pow(base[1], exponent))


def power_standard(base: torch.Tensor,
                   exponent: float | int | torch.Tensor) -> torch.Tensor:
    """
    Raises a bicomplex tensor to a power in standard form.

    Args:
        base: Bicomplex tensor of shape (..., 4)
        exponent: Real scalar, integer, or tensor

    Returns:
        Bicomplex tensor of shape (..., 4) representing base^exponent
    """
    if not is_bicomplex(base):
        raise ValueError("Base must be a bicomplex tensor")


    idempotent = to_idempotent(base)
    result_idempotent = power_idempotent(idempotent, exponent)
    return from_idempotent(result_idempotent)


def square_idempotent(tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the square of a bicomplex tensor in idempotent form.

    Optimized version of power_idempotent with exponent=2.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (tensor[0] ** 2, tensor[1] ** 2)


def sqrt_idempotent(tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the square root of a bicomplex tensor in idempotent form.

    For z = (e1, e2), sqrt(z) = (sqrt(e1), sqrt(e2))

    Note: Returns principal square root. For complex e1, e2, torch.sqrt handles
    the complex square root automatically.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.sqrt(tensor[0]), torch.sqrt(tensor[1]))


def sqrt_standard(bicomplex_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the square root of a bicomplex tensor in standard form.
    """
    if not is_bicomplex(bicomplex_tensor):
        raise ValueError("Input must be a bicomplex tensor")

    idempotent = to_idempotent(bicomplex_tensor)
    result_idempotent = sqrt_idempotent(idempotent)
    return from_idempotent(result_idempotent)


def reciprocal_idempotent(
    tensor: tuple[torch.Tensor, torch.Tensor],
    zero_divisor_handling: str = 'error',
    eps: float = 1e-10,
    regularization: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the reciprocal (1/z) of a bicomplex tensor in idempotent form.

    Equivalent to inverse_idempotent, provided for API completeness.

    Args:
        tensor: Bicomplex tensor in idempotent form (e1, e2)
        zero_divisor_handling: Strategy for handling zero divisors
            ('error', 'warn', 'regularize', or 'ignore')
        eps: Threshold for detecting zero divisors
        regularization: Value added to regularize near-zero components

    Returns:
        Tuple (1/e1, 1/e2) representing 1/z in idempotent form
    """
    return inverse_idempotent(tensor, zero_divisor_handling, eps, regularization)


def bicomplex_power_idempotent(base: tuple[torch.Tensor, torch.Tensor],
                               exponent: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Raises a bicomplex tensor to a bicomplex power: z^w where both z and w are bicomplex.

    Uses the identity: z^w = exp(w * log(z))

    Args:
        base: Bicomplex tensor in idempotent form (z)
        exponent: Bicomplex tensor in idempotent form (w)

    Returns:
        Tuple representing z^w in idempotent form
    """
    if not is_idempotent(base):
        raise ValueError("Base must be a bicomplex tensor in idempotent form")
    if not is_idempotent(exponent):
        raise ValueError("Exponent must be a bicomplex tensor in idempotent form")

    # z^w = exp(w * log(z))
    log_base = log_idempotent(base)
    w_log_z = multiply_idempotent(exponent, log_base)
    return exp_idempotent(w_log_z)