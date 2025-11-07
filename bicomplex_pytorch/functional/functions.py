"""
Functional API for bicomplex operations.

This module provides stateless functions for bicomplex tensor operations,
similar to torch.nn.functional. These functions don't maintain state and
require all parameters (weights, biases) to be passed explicitly.

All operations work on bicomplex tensors in idempotent form: tuple[torch.Tensor, torch.Tensor]
where each element is a complex PyTorch tensor representing (e1, e2) components.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List


# ============================================================================
# Linear Operations
# ============================================================================

def bicomplex_linear(
        input: Tuple[torch.Tensor, torch.Tensor],
        weight: Tuple[torch.Tensor, torch.Tensor],
        bias: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a bicomplex linear transformation to the incoming data.

    Args:
        input: Bicomplex tensor, each component shape (..., in_features)
        weight: Bicomplex weight tensor, each component shape (out_features, in_features)
        bias: Optional bicomplex bias tensor, each component shape (out_features,)

    Returns:
        Bicomplex output tensor, each component shape (..., out_features)
    """
    from bicomplex_pytorch.core.tensor_ops import matmul, is_idempotent
    from bicomplex_pytorch.core.arithmetic import add

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")

    # Transpose weight for linear operation: input @ weight.T
    weight_t = (weight[0].t(), weight[1].t())
    output = matmul(input, weight_t)

    if bias is not None:
        if not is_idempotent(bias):
            raise ValueError("Bias must be a bicomplex tensor in idempotent form")
        output = add(output, bias)

    return output


# ============================================================================
# Convolutional Operations
# ============================================================================

def bicomplex_conv1d(
        input: Tuple[torch.Tensor, torch.Tensor],
        weight: Tuple[torch.Tensor, torch.Tensor],
        bias: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        stride: int = 1,
        padding: Union[int, str] = 0,
        dilation: int = 1,
        groups: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a 1D bicomplex convolution.

    Each idempotent component is convolved independently.

    Args:
        input: Bicomplex tensor, each component shape (batch, in_channels, length)
        weight: Bicomplex weight, each component shape (out_channels, in_channels//groups, kernel_size)
        bias: Optional bicomplex bias, each component shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding added to both sides
        dilation: Spacing between kernel elements
        groups: Number of blocked connections

    Returns:
        Bicomplex output, each component shape (batch, out_channels, length_out)
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")

    # Convolve each idempotent component independently
    e1_out = F.conv1d(
        input[0],
        weight[0],
        bias=bias[0] if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    e2_out = F.conv1d(
        input[1],
        weight[1],
        bias=bias[1] if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    return (e1_out, e2_out)


def bicomplex_conv2d(
        input: Tuple[torch.Tensor, torch.Tensor],
        weight: Tuple[torch.Tensor, torch.Tensor],
        bias: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a 2D bicomplex convolution.

    Args:
        input: Bicomplex tensor, each component shape (batch, in_channels, height, width)
        weight: Bicomplex weight, each component shape (out_channels, in_channels//groups, kH, kW)
        bias: Optional bicomplex bias, each component shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding added to all sides
        dilation: Spacing between kernel elements
        groups: Number of blocked connections

    Returns:
        Bicomplex output, each component shape (batch, out_channels, height_out, width_out)
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")

    e1_out = F.conv2d(
        input[0],
        weight[0],
        bias=bias[0] if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    e2_out = F.conv2d(
        input[1],
        weight[1],
        bias=bias[1] if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    return (e1_out, e2_out)


def bicomplex_conv3d(
        input: Tuple[torch.Tensor, torch.Tensor],
        weight: Tuple[torch.Tensor, torch.Tensor],
        bias: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int], str] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a 3D bicomplex convolution.

    Args:
        input: Bicomplex tensor, each component shape (batch, in_channels, depth, height, width)
        weight: Bicomplex weight, each component shape (out_channels, in_channels//groups, kD, kH, kW)
        bias: Optional bicomplex bias, each component shape (out_channels,)
        stride: Stride of the convolution
        padding: Padding added to all sides
        dilation: Spacing between kernel elements
        groups: Number of blocked connections

    Returns:
        Bicomplex output, each component shape (batch, out_channels, depth_out, height_out, width_out)
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")

    e1_out = F.conv3d(
        input[0],
        weight[0],
        bias=bias[0] if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    e2_out = F.conv3d(
        input[1],
        weight[1],
        bias=bias[1] if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    return (e1_out, e2_out)


# ============================================================================
# Pooling Operations
# ============================================================================

def bicomplex_max_pool1d(
        input: Tuple[torch.Tensor, torch.Tensor],
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Applies 1D max pooling using bicomplex modulus.

    The maximum is determined by the bicomplex modulus: |z|² = |e1|² + |e2|²

    Args:
        input: Bicomplex tensor, each component shape (batch, channels, length)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        padding: Padding added to both sides
        dilation: Spacing between pooling elements
        return_indices: If True, return indices along with outputs
        ceil_mode: Use ceil instead of floor for output shape

    Returns:
        Bicomplex pooled tensor or (pooled_tensor, indices) if return_indices=True
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent
    from bicomplex_pytorch.core.arithmetic import modulus_squared

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Calculate modulus squared for pooling criterion: |e1|² + |e2|²
    mod_sq = modulus_squared(input)

    # Get indices of maximum modulus
    if return_indices:
        _, indices = F.max_pool1d(
            mod_sq,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=True,
            ceil_mode=ceil_mode
        )
    else:
        pooled_mod = F.max_pool1d(
            mod_sq,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=False,
            ceil_mode=ceil_mode
        )
        # Get indices for gathering
        _, indices = F.max_pool1d(
            mod_sq,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=True,
            ceil_mode=ceil_mode
        )

    # Gather the actual bicomplex values at max indices
    # Flatten spatial dims for gathering
    batch, channels, length = input[0].shape
    stride_val = stride if stride is not None else kernel_size

    # Use unfold to get all candidates
    input_e1_unfolded = F.unfold(
        input[0].unsqueeze(-1),  # Add dummy dimension
        kernel_size=(kernel_size, 1),
        stride=(stride_val, 1),
        padding=(padding, 0),
        dilation=(dilation, 1)
    )
    input_e2_unfolded = F.unfold(
        input[1].unsqueeze(-1),
        kernel_size=(kernel_size, 1),
        stride=(stride_val, 1),
        padding=(padding, 0),
        dilation=(dilation, 1)
    )

    # Gather using indices
    batch_size, num_channels, out_length = indices.shape
    flat_indices = indices.view(batch_size, num_channels * out_length)

    e1_out = torch.gather(input_e1_unfolded, 1, flat_indices.unsqueeze(1).expand(-1, kernel_size, -1))
    e2_out = torch.gather(input_e2_unfolded, 1, flat_indices.unsqueeze(1).expand(-1, kernel_size, -1))

    # Reshape back
    e1_out = e1_out[:, 0, :].view(batch_size, num_channels, out_length)
    e2_out = e2_out[:, 0, :].view(batch_size, num_channels, out_length)

    output = (e1_out, e2_out)

    if return_indices:
        return output, (indices, indices)
    return output


def bicomplex_max_pool2d(
        input: Tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Applies 2D max pooling using bicomplex modulus.

    Args:
        input: Bicomplex tensor, each component shape (batch, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        padding: Padding added to all sides
        dilation: Spacing between pooling elements
        return_indices: If True, return indices along with outputs
        ceil_mode: Use ceil instead of floor for output shape

    Returns:
        Bicomplex pooled tensor or (pooled_tensor, indices) if return_indices=True
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent
    from bicomplex_pytorch.core.arithmetic import modulus_squared

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Calculate modulus squared
    mod_sq = modulus_squared(input)

    # Pool based on modulus and get indices
    _, indices = F.max_pool2d(
        mod_sq,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=True,
        ceil_mode=ceil_mode
    )

    # Flatten for gathering
    batch, channels, h_in, w_in = input[0].shape
    _, _, h_out, w_out = indices.shape

    # Flatten spatial dimensions
    e1_flat = input[0].view(batch, channels, -1)
    e2_flat = input[1].view(batch, channels, -1)
    indices_flat = indices.view(batch, channels, -1)

    # Gather values
    e1_out = torch.gather(e1_flat, 2, indices_flat)
    e2_out = torch.gather(e2_flat, 2, indices_flat)

    # Reshape back
    e1_out = e1_out.view(batch, channels, h_out, w_out)
    e2_out = e2_out.view(batch, channels, h_out, w_out)

    output = (e1_out, e2_out)

    if return_indices:
        return output, (indices, indices)
    return output


def bicomplex_avg_pool1d(
        input: Tuple[torch.Tensor, torch.Tensor],
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 1D average pooling to bicomplex tensor.

    Each component is averaged independently.

    Args:
        input: Bicomplex tensor, each component shape (batch, channels, length)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        padding: Padding added to both sides
        ceil_mode: Use ceil instead of floor for output shape
        count_include_pad: Include padding in the averaging calculation

    Returns:
        Bicomplex pooled tensor, each component shape (batch, channels, length_out)
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1_out = F.avg_pool1d(
        input[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )

    e2_out = F.avg_pool1d(
        input[1],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )

    return (e1_out, e2_out)


def bicomplex_avg_pool2d(
        input: Tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 2D average pooling to bicomplex tensor.

    Args:
        input: Bicomplex tensor, each component shape (batch, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        padding: Padding added to all sides
        ceil_mode: Use ceil instead of floor for output shape
        count_include_pad: Include padding in the averaging calculation

    Returns:
        Bicomplex pooled tensor, each component shape (batch, channels, height_out, width_out)
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1_out = F.avg_pool2d(
        input[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )

    e2_out = F.avg_pool2d(
        input[1],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )

    return (e1_out, e2_out)


def bicomplex_avg_pool3d(
        input: Tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 3D average pooling to bicomplex tensor.

    Args:
        input: Bicomplex tensor, each component shape (batch, channels, depth, height, width)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        padding: Padding added to all sides
        ceil_mode: Use ceil instead of floor for output shape
        count_include_pad: Include padding in the averaging calculation

    Returns:
        Bicomplex pooled tensor, each component shape (batch, channels, depth_out, height_out, width_out)
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1_out = F.avg_pool3d(
        input[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )

    e2_out = F.avg_pool3d(
        input[1],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )

    return (e1_out, e2_out)


# ============================================================================
# Activation Functions
# ============================================================================

def bicomplex_relu(
        input: Tuple[torch.Tensor, torch.Tensor],
        inplace: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies ReLU activation to bicomplex numbers component-wise.

    Applies ReLU independently to each idempotent component.

    Args:
        input: Bicomplex tensor in idempotent form
        inplace: If True, do the operation in-place

    Returns:
        Activated bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if inplace:
        return (F.relu(input[0], inplace=True), F.relu(input[1], inplace=True))
    return (F.relu(input[0]), F.relu(input[1]))


def bicomplex_leaky_relu(
        input: Tuple[torch.Tensor, torch.Tensor],
        negative_slope: float = 0.01,
        inplace: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Leaky ReLU activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        negative_slope: Controls the angle of the negative slope
        inplace: If True, do the operation in-place

    Returns:
        Activated bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if inplace:
        return (
            F.leaky_relu(input[0], negative_slope=negative_slope, inplace=True),
            F.leaky_relu(input[1], negative_slope=negative_slope, inplace=True)
        )
    return (
        F.leaky_relu(input[0], negative_slope=negative_slope),
        F.leaky_relu(input[1], negative_slope=negative_slope)
    )


def bicomplex_elu(
        input: Tuple[torch.Tensor, torch.Tensor],
        alpha: float = 1.0,
        inplace: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies ELU activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        alpha: The α value for the ELU formulation
        inplace: If True, do the operation in-place

    Returns:
        Activated bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if inplace:
        return (
            F.elu(input[0], alpha=alpha, inplace=True),
            F.elu(input[1], alpha=alpha, inplace=True)
        )
    return (F.elu(input[0], alpha=alpha), F.elu(input[1], alpha=alpha))


def bicomplex_gelu(
        input: Tuple[torch.Tensor, torch.Tensor],
        approximate: str = 'none'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies GELU activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        approximate: The approximation method ('none' or 'tanh')

    Returns:
        Activated bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (
        F.gelu(input[0], approximate=approximate),
        F.gelu(input[1], approximate=approximate)
    )


def bicomplex_silu(
        input: Tuple[torch.Tensor, torch.Tensor],
        inplace: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies SiLU (Swish) activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form
        inplace: If True, do the operation in-place

    Returns:
        Activated bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if inplace:
        return (F.silu(input[0], inplace=True), F.silu(input[1], inplace=True))
    return (F.silu(input[0]), F.silu(input[1]))


def bicomplex_tanh(
        input: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Tanh activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Activated bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.tanh(input[0]), torch.tanh(input[1]))


def bicomplex_sigmoid(
        input: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Sigmoid activation component-wise.

    Args:
        input: Bicomplex tensor in idempotent form

    Returns:
        Activated bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.sigmoid(input[0]), torch.sigmoid(input[1]))


def bicomplex_softmax(
        input: Tuple[torch.Tensor, torch.Tensor],
        dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Softmax function based on bicomplex modulus.

    The softmax is computed using the modulus of bicomplex numbers as logits.

    Args:
        input: Bicomplex tensor in idempotent form
        dim: Dimension along which softmax is computed

    Returns:
        Bicomplex tensor with softmax applied
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent
    from bicomplex_pytorch.core.arithmetic import modulus

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Compute softmax weights based on modulus
    mod = modulus(input)
    weights = F.softmax(mod, dim=dim)

    # Apply weights to each component
    # Expand weights to match input shape
    weights_expanded = weights.unsqueeze(-1) if weights.dim() < input[0].dim() else weights

    return (input[0] * weights_expanded, input[1] * weights_expanded)


def bicomplex_log_softmax(
        input: Tuple[torch.Tensor, torch.Tensor],
        dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Log-Softmax function based on bicomplex modulus.

    Args:
        input: Bicomplex tensor in idempotent form
        dim: Dimension along which log_softmax is computed

    Returns:
        Bicomplex tensor with log_softmax applied
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent
    from bicomplex_pytorch.core.arithmetic import modulus

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Compute log_softmax weights based on modulus
    mod = modulus(input)
    log_weights = F.log_softmax(mod, dim=dim)

    # Apply to each component
    log_weights_expanded = log_weights.unsqueeze(-1) if log_weights.dim() < input[0].dim() else log_weights

    return (input[0] + log_weights_expanded, input[1] + log_weights_expanded)


# ============================================================================
# Normalization
# ============================================================================

def bicomplex_batch_norm(
        input: Tuple[torch.Tensor, torch.Tensor],
        running_mean: Optional[Tuple[torch.Tensor, torch.Tensor]],
        running_var: Optional[Tuple[torch.Tensor, torch.Tensor]],
        weight: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        bias: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Batch Normalization to each idempotent component independently.

    Args:
        input: Bicomplex tensor in idempotent form
        running_mean: Running mean for each component
        running_var: Running variance for each component
        weight: Learnable scale parameter for each component
        bias: Learnable shift parameter for each component
        training: If True, use batch statistics; else use running statistics
        momentum: Momentum for running statistics
        eps: Small constant for numerical stability

    Returns:
        Normalized bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1_out = F.batch_norm(
        input[0],
        running_mean=running_mean[0] if running_mean is not None else None,
        running_var=running_var[0] if running_var is not None else None,
        weight=weight[0] if weight is not None else None,
        bias=bias[0] if bias is not None else None,
        training=training,
        momentum=momentum,
        eps=eps
    )

    e2_out = F.batch_norm(
        input[1],
        running_mean=running_mean[1] if running_mean is not None else None,
        running_var=running_var[1] if running_var is not None else None,
        weight=weight[1] if weight is not None else None,
        bias=bias[1] if bias is not None else None,
        training=training,
        momentum=momentum,
        eps=eps
    )

    return (e1_out, e2_out)


def bicomplex_layer_norm(
        input: Tuple[torch.Tensor, torch.Tensor],
        normalized_shape: Union[int, List[int], torch.Size],
        weight: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        bias: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Layer Normalization to each idempotent component independently.

    Args:
        input: Bicomplex tensor in idempotent form
        normalized_shape: Input shape from an expected input
        weight: Learnable scale parameter for each component
        bias: Learnable shift parameter for each component
        eps: Small constant for numerical stability

    Returns:
        Normalized bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1_out = F.layer_norm(
        input[0],
        normalized_shape=normalized_shape,
        weight=weight[0] if weight is not None else None,
        bias=bias[0] if bias is not None else None,
        eps=eps
    )

    e2_out = F.layer_norm(
        input[1],
        normalized_shape=normalized_shape,
        weight=weight[1] if weight is not None else None,
        bias=bias[1] if bias is not None else None,
        eps=eps
    )

    return (e1_out, e2_out)


# ============================================================================
# Dropout
# ============================================================================

def bicomplex_dropout(
        input: Tuple[torch.Tensor, torch.Tensor],
        p: float = 0.5,
        training: bool = True,
        inplace: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies dropout to bicomplex tensor.

    Uses the same dropout mask for both idempotent components to preserve
    the bicomplex structure.

    Args:
        input: Bicomplex tensor in idempotent form
        p: Probability of an element to be zeroed
        training: If True, apply dropout; else return input unchanged
        inplace: If True, do the operation in-place

    Returns:
        Bicomplex tensor with dropout applied
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if not training or p == 0:
        return input

    # Generate a single mask based on the first component
    mask = torch.rand_like(input[0].real if input[0].is_complex() else input[0]) > p
    mask = mask / (1 - p)  # Scale to maintain expected value

    # Apply same mask to both components
    if inplace:
        input[0].mul_(mask)
        input[1].mul_(mask)
        return input
    else:
        return (input[0] * mask, input[1] * mask)


def bicomplex_dropout2d(
        input: Tuple[torch.Tensor, torch.Tensor],
        p: float = 0.5,
        training: bool = True,
        inplace: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 2D dropout to bicomplex tensor (drops entire channels).

    Args:
        input: Bicomplex tensor in idempotent form, shape (batch, channels, height, width)
        p: Probability of a channel to be zeroed
        training: If True, apply dropout; else return input unchanged
        inplace: If True, do the operation in-place

    Returns:
        Bicomplex tensor with dropout applied
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if not training or p == 0:
        return input

    # Generate channel-wise mask
    batch, channels = input[0].shape[:2]
    mask_shape = (batch, channels) + (1,) * (input[0].dim() - 2)
    mask = torch.rand(mask_shape, device=input[0].device, dtype=input[0].dtype) > p
    mask = mask / (1 - p)

    if inplace:
        input[0].mul_(mask)
        input[1].mul_(mask)
        return input
    else:
        return (input[0] * mask, input[1] * mask)


# ============================================================================
# Loss Functions
# ============================================================================

def bicomplex_mse_loss(
        input: Tuple[torch.Tensor, torch.Tensor],
        target: Tuple[torch.Tensor, torch.Tensor],
        reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes Mean Squared Error loss for bicomplex tensors.

    Loss = MSE(e1_input, e1_target) + MSE(e2_input, e2_target)

    Args:
        input: Predicted bicomplex tensor
        target: Target bicomplex tensor
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        Scalar loss value
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(target):
        raise ValueError("Target must be a bicomplex tensor in idempotent form")

    loss_e1 = F.mse_loss(input[0], target[0], reduction=reduction)
    loss_e2 = F.mse_loss(input[1], target[1], reduction=reduction)

    if reduction == 'none':
        return loss_e1 + loss_e2
    else:
        return loss_e1 + loss_e2


def bicomplex_l1_loss(
        input: Tuple[torch.Tensor, torch.Tensor],
        target: Tuple[torch.Tensor, torch.Tensor],
        reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes L1 loss for bicomplex tensors.

    Args:
        input: Predicted bicomplex tensor
        target: Target bicomplex tensor
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        Scalar loss value
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(target):
        raise ValueError("Target must be a bicomplex tensor in idempotent form")

    loss_e1 = F.l1_loss(input[0], target[0], reduction=reduction)
    loss_e2 = F.l1_loss(input[1], target[1], reduction=reduction)

    if reduction == 'none':
        return loss_e1 + loss_e2
    else:
        return loss_e1 + loss_e2


def bicomplex_smooth_l1_loss(
        input: Tuple[torch.Tensor, torch.Tensor],
        target: Tuple[torch.Tensor, torch.Tensor],
        reduction: str = 'mean',
        beta: float = 1.0
) -> torch.Tensor:
    """
    Computes Smooth L1 loss for bicomplex tensors.

    Args:
        input: Predicted bicomplex tensor
        target: Target bicomplex tensor
        reduction: 'none' | 'mean' | 'sum'
        beta: Threshold at which to change between L1 and L2 loss

    Returns:
        Scalar loss value
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(target):
        raise ValueError("Target must be a bicomplex tensor in idempotent form")

    loss_e1 = F.smooth_l1_loss(input[0], target[0], reduction=reduction, beta=beta)
    loss_e2 = F.smooth_l1_loss(input[1], target[1], reduction=reduction, beta=beta)

    if reduction == 'none':
        return loss_e1 + loss_e2
    else:
        return loss_e1 + loss_e2


def bicomplex_cross_entropy(
        input: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        ignore_index: int = -100
) -> torch.Tensor:
    """
    Computes Cross Entropy loss using bicomplex modulus as logits.

    Args:
        input: Bicomplex tensor (batch, num_classes)
        target: Target class indices (batch,)
        weight: Manual rescaling weight for each class
        reduction: 'none' | 'mean' | 'sum'
        ignore_index: Specifies a target value that is ignored

    Returns:
        Scalar loss value
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent
    from bicomplex_pytorch.core.arithmetic import modulus

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Use modulus as logits
    logits = modulus(input)

    return F.cross_entropy(
        logits,
        target,
        weight=weight,
        reduction=reduction,
        ignore_index=ignore_index
    )


def bicomplex_nll_loss(
        input: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        ignore_index: int = -100
) -> torch.Tensor:
    """
    Computes Negative Log Likelihood loss using bicomplex modulus.

    Args:
        input: Bicomplex tensor (batch, num_classes) - should be log-probabilities
        target: Target class indices (batch,)
        weight: Manual rescaling weight for each class
        reduction: 'none' | 'mean' | 'sum'
        ignore_index: Specifies a target value that is ignored

    Returns:
        Scalar loss value
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent
    from bicomplex_pytorch.core.arithmetic import modulus

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Use modulus for loss computation
    log_probs = modulus(input)

    return F.nll_loss(
        log_probs,
        target,
        weight=weight,
        reduction=reduction,
        ignore_index=ignore_index
    )


# ============================================================================
# Interpolation
# ============================================================================

def bicomplex_interpolate(
        input: Tuple[torch.Tensor, torch.Tensor],
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
        mode: str = 'nearest',
        align_corners: Optional[bool] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Down/up samples bicomplex tensor to given size or by scale_factor.

    Args:
        input: Bicomplex tensor in idempotent form
        size: Output spatial size
        scale_factor: Multiplier for spatial size
        mode: Algorithm used for upsampling
        align_corners: If True, corner pixels are aligned

    Returns:
        Interpolated bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1_out = F.interpolate(
        input[0],
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners
    )

    e2_out = F.interpolate(
        input[1],
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners
    )

    return (e1_out, e2_out)


# ============================================================================
# Padding
# ============================================================================

def bicomplex_pad(
        input: Tuple[torch.Tensor, torch.Tensor],
        pad: Tuple[int, ...],
        mode: str = 'constant',
        value: float = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads bicomplex tensor.

    Args:
        input: Bicomplex tensor in idempotent form
        pad: Padding sizes (m_begin, m_end, n_begin, n_end, ...)
        mode: 'constant' | 'reflect' | 'replicate' | 'circular'
        value: Fill value for constant padding

    Returns:
        Padded bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1_out = F.pad(input[0], pad, mode=mode, value=value)
    e2_out = F.pad(input[1], pad, mode=mode, value=value)

    return (e1_out, e2_out)


# ============================================================================
# Utility Functions
# ============================================================================

def bicomplex_normalize(
        input: Tuple[torch.Tensor, torch.Tensor],
        p: float = 2.0,
        dim: int = -1,
        eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes bicomplex tensor using its modulus.

    Args:
        input: Bicomplex tensor in idempotent form
        p: Norm order
        dim: Dimension along which to normalize
        eps: Small value to avoid division by zero

    Returns:
        Normalized bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent
    from bicomplex_pytorch.core.arithmetic import modulus

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Compute modulus for normalization
    mod = modulus(input)
    norm = torch.norm(mod, p=p, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)

    # Normalize each component
    return (input[0] / norm, input[1] / norm)


def bicomplex_flatten(
        input: Tuple[torch.Tensor, torch.Tensor],
        start_dim: int = 0,
        end_dim: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flattens bicomplex tensor dimensions.

    Args:
        input: Bicomplex tensor in idempotent form
        start_dim: First dimension to flatten
        end_dim: Last dimension to flatten

    Returns:
        Flattened bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import flatten, is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return flatten(input, start_dim=start_dim, end_dim=end_dim)


def bicomplex_unfold(
        input: Tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        stride: Union[int, Tuple[int, int]] = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts sliding local blocks from bicomplex tensor.

    Args:
        input: Bicomplex tensor in idempotent form
        kernel_size: Size of the sliding blocks
        dilation: Spacing between kernel elements
        padding: Implicit zero padding
        stride: Stride of the sliding blocks

    Returns:
        Unfolded bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1_out = F.unfold(
        input[0],
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride
    )

    e2_out = F.unfold(
        input[1],
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride
    )

    return (e1_out, e2_out)


def bicomplex_fold(
        input: Tuple[torch.Tensor, torch.Tensor],
        output_size: Tuple[int, int],
        kernel_size: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        stride: Union[int, Tuple[int, int]] = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combines sliding local blocks into bicomplex tensor.

    Args:
        input: Bicomplex tensor in idempotent form
        output_size: Shape of the output
        kernel_size: Size of the sliding blocks
        dilation: Spacing between kernel elements
        padding: Implicit zero padding
        stride: Stride of the sliding blocks

    Returns:
        Folded bicomplex tensor
    """
    from bicomplex_pytorch.core.tensor_ops import is_idempotent

    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1_out = F.fold(
        input[0],
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride
    )

    e2_out = F.fold(
        input[1],
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride
    )

    return (e1_out, e2_out)