"""
This module provides utilities for arithmetic operations on bicomplex numbers
represented as tensors.

A bicomplex number is represented in the form:
    z = a + b i + c j + d i j
where i and j are imaginary units such that i² = j² = -1 and i j = j i.

The module includes functions for creating, converting, extracting components,
and performing transformations on bicomplex tensors.
"""

import torch
from typing import Tuple
import numpy

from representations import is_bicomplex, is_idempotent, to_idempotent, from_idempotent


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


def inverse_idempotent(tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the multiplicative inverse of a bicomplex tensor in idempotent form.

    For (e1, e2), the inverse is (1/e1, 1/e2).
    Note: This will raise errors for zero divisors where e1=0 or e2=0.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1, e2 = tensor
    return (1.0 / e1, 1.0 / e2)


def divide_idempotent(numerator: tuple[torch.Tensor, torch.Tensor],
                      denominator: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Divides two bicomplex tensors in idempotent form.

    Division is component-wise: (e1, e2) / (e1', e2') = (e1/e1', e2/e2')
    """
    if not is_idempotent(numerator):
        raise ValueError("Numerator must be a bicomplex tensor in idempotent form")
    if not is_idempotent(denominator):
        raise ValueError("Denominator must be a bicomplex tensor in idempotent form")

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


def log_idempotent(tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the natural logarithm of a bicomplex tensor in idempotent form.

    For z = (e1, e2), log(z) = (log(e1), log(e2))

    Note: Both e1 and e2 should be complex tensors. If they're real, ensure they're positive
    to avoid NaN results, or convert to complex first.

    Args:
        tensor: Bicomplex tensor in idempotent form (e1, e2)

    Returns:
        Tuple (log(e1), log(e2)) representing log(z) in idempotent form
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

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


def reciprocal_idempotent(tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the reciprocal (1/z) of a bicomplex tensor in idempotent form.

    Equivalent to inverse_idempotent, provided for completeness.
    """
    return inverse_idempotent(tensor)


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