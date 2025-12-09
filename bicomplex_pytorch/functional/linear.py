"""
Linear transformation functions for bicomplex neural networks.

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
from ..core.tensor_ops import matmul


def bicomplex_linear(
        input: tuple[torch.Tensor, torch.Tensor],
        weight: tuple[torch.Tensor, torch.Tensor],
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a linear transformation to bicomplex input in idempotent form.

    Computes: output = input @ weight^T + bias

    In idempotent form, the multiplication is component-wise:
        output_e1 = input_e1 @ weight_e1^T + bias_e1
        output_e2 = input_e2 @ weight_e2^T + bias_e2

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (..., in_features) for each component
        weight: Bicomplex weight tensor in idempotent form
                Shape: (out_features, in_features) for each component
        bias: Optional bicomplex bias tensor in idempotent form
              Shape: (out_features,) for each component

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (..., out_features) for each component

    Raises:
        ValueError: If inputs are not valid bicomplex tensors in idempotent form

    Example:
        >>> input = (torch.randn(32, 128), torch.randn(32, 128))  # batch_size=32, in_features=128
        >>> weight = (torch.randn(64, 128), torch.randn(64, 128))  # out_features=64
        >>> bias = (torch.randn(64), torch.randn(64))
        >>> output = bicomplex_linear(input, weight, bias)
        >>> output[0].shape, output[1].shape
        (torch.Size([32, 64]), torch.Size([32, 64]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    # Perform linear transformation for e1 component
    output_e1 = torch.nn.functional.linear(input[0], weight[0], None)

    # Perform linear transformation for e2 component
    output_e2 = torch.nn.functional.linear(input[1], weight[1], None)

    # Add bias if provided
    if bias is not None:
        output_e1 = output_e1 + bias[0]
        output_e2 = output_e2 + bias[1]

    return (output_e1, output_e2)


def bicomplex_bilinear(
        input1: tuple[torch.Tensor, torch.Tensor],
        input2: tuple[torch.Tensor, torch.Tensor],
        weight: tuple[torch.Tensor, torch.Tensor],
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a bilinear transformation to bicomplex inputs in idempotent form.

    Computes: output = input1 @ weight @ input2^T + bias

    In idempotent form:
        output_e1 = input1_e1 @ weight_e1 @ input2_e1^T + bias_e1
        output_e2 = input1_e2 @ weight_e2 @ input2_e2^T + bias_e2

    Args:
        input1: First bicomplex input tensor in idempotent form
                Shape: (batch, in1_features) for each component
        input2: Second bicomplex input tensor in idempotent form
                Shape: (batch, in2_features) for each component
        weight: Bicomplex weight tensor in idempotent form
                Shape: (out_features, in1_features, in2_features) for each component
        bias: Optional bicomplex bias tensor in idempotent form
              Shape: (out_features,) for each component

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (batch, out_features) for each component

    Example:
        >>> input1 = (torch.randn(32, 64), torch.randn(32, 64))
        >>> input2 = (torch.randn(32, 48), torch.randn(32, 48))
        >>> weight = (torch.randn(128, 64, 48), torch.randn(128, 64, 48))
        >>> bias = (torch.randn(128), torch.randn(128))
        >>> output = bicomplex_bilinear(input1, input2, weight, bias)
        >>> output[0].shape, output[1].shape
        (torch.Size([32, 128]), torch.Size([32, 128]))
    """
    if not is_idempotent(input1):
        raise ValueError("First input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(input2):
        raise ValueError("Second input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    # Perform bilinear transformation for e1 component
    output_e1 = torch.nn.functional.bilinear(input1[0], input2[0], weight[0], None)

    # Perform bilinear transformation for e2 component
    output_e2 = torch.nn.functional.bilinear(input1[1], input2[1], weight[1], None)

    # Add bias if provided
    if bias is not None:
        output_e1 = output_e1 + bias[0]
        output_e2 = output_e2 + bias[1]

    return (output_e1, output_e2)


def bicomplex_matmul(
        input: tuple[torch.Tensor, torch.Tensor],
        other: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Matrix multiplication for bicomplex tensors in idempotent form.

    This is an alias for the matmul function from tensor_ops,
    provided here for completeness in the functional API.

    Args:
        input: First bicomplex tensor, each component shape (..., M, N)
        other: Second bicomplex tensor, each component shape (..., N, P)

    Returns:
        Tuple of tensors with shape (..., M, P)

    Example:
        >>> a = (torch.randn(32, 10, 20), torch.randn(32, 10, 20))
        >>> b = (torch.randn(32, 20, 15), torch.randn(32, 20, 15))
        >>> c = bicomplex_matmul(a, b)
        >>> c[0].shape, c[1].shape
        (torch.Size([32, 10, 15]), torch.Size([32, 10, 15]))
    """
    return matmul(input, other)


def bicomplex_addmm(
        bias: tuple[torch.Tensor, torch.Tensor],
        input: tuple[torch.Tensor, torch.Tensor],
        mat: tuple[torch.Tensor, torch.Tensor],
        beta: float = 1.0,
        alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a matrix multiplication and addition for bicomplex tensors.

    Computes: output = beta * bias + alpha * (input @ mat)

    In idempotent form:
        output_e1 = beta * bias_e1 + alpha * (input_e1 @ mat_e1)
        output_e2 = beta * bias_e2 + alpha * (input_e2 @ mat_e2)

    Args:
        bias: Bicomplex bias tensor to be added
              Shape: (N, P) for each component
        input: Bicomplex input matrix
               Shape: (N, M) for each component
        mat: Bicomplex matrix to multiply with input
             Shape: (M, P) for each component
        beta: Multiplier for bias (default: 1.0)
        alpha: Multiplier for input @ mat (default: 1.0)

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (N, P) for each component

    Example:
        >>> bias = (torch.randn(10, 20), torch.randn(10, 20))
        >>> input = (torch.randn(10, 15), torch.randn(10, 15))
        >>> mat = (torch.randn(15, 20), torch.randn(15, 20))
        >>> output = bicomplex_addmm(bias, input, mat, beta=0.5, alpha=2.0)
    """
    if not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(mat):
        raise ValueError("Mat must be a bicomplex tensor in idempotent form")

    output_e1 = torch.addmm(bias[0], input[0], mat[0], beta=beta, alpha=alpha)
    output_e2 = torch.addmm(bias[1], input[1], mat[1], beta=beta, alpha=alpha)

    return (output_e1, output_e2)


def bicomplex_addmv(
        bias: tuple[torch.Tensor, torch.Tensor],
        mat: tuple[torch.Tensor, torch.Tensor],
        vec: tuple[torch.Tensor, torch.Tensor],
        beta: float = 1.0,
        alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a matrix-vector multiplication and addition for bicomplex tensors.

    Computes: output = beta * bias + alpha * (mat @ vec)

    Args:
        bias: Bicomplex bias vector
              Shape: (N,) for each component
        mat: Bicomplex matrix
             Shape: (N, M) for each component
        vec: Bicomplex vector
             Shape: (M,) for each component
        beta: Multiplier for bias (default: 1.0)
        alpha: Multiplier for mat @ vec (default: 1.0)

    Returns:
        Bicomplex output vector in idempotent form
        Shape: (N,) for each component
    """
    if not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")
    if not is_idempotent(mat):
        raise ValueError("Mat must be a bicomplex tensor in idempotent form")
    if not is_idempotent(vec):
        raise ValueError("Vec must be a bicomplex tensor in idempotent form")

    output_e1 = torch.addmv(bias[0], mat[0], vec[0], beta=beta, alpha=alpha)
    output_e2 = torch.addmv(bias[1], mat[1], vec[1], beta=beta, alpha=alpha)

    return (output_e1, output_e2)


def bicomplex_addr(
        input: tuple[torch.Tensor, torch.Tensor],
        vec1: tuple[torch.Tensor, torch.Tensor],
        vec2: tuple[torch.Tensor, torch.Tensor],
        beta: float = 1.0,
        alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs an outer product of vectors and adds it to a matrix.

    Computes: output = beta * input + alpha * (vec1 ⊗ vec2)

    Args:
        input: Bicomplex input matrix
               Shape: (N, M) for each component
        vec1: First bicomplex vector
              Shape: (N,) for each component
        vec2: Second bicomplex vector
              Shape: (M,) for each component
        beta: Multiplier for input (default: 1.0)
        alpha: Multiplier for outer product (default: 1.0)

    Returns:
        Bicomplex output matrix in idempotent form
        Shape: (N, M) for each component
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(vec1):
        raise ValueError("Vec1 must be a bicomplex tensor in idempotent form")
    if not is_idempotent(vec2):
        raise ValueError("Vec2 must be a bicomplex tensor in idempotent form")

    output_e1 = torch.addr(input[0], vec1[0], vec2[0], beta=beta, alpha=alpha)
    output_e2 = torch.addr(input[1], vec1[1], vec2[1], beta=beta, alpha=alpha)

    return (output_e1, output_e2)