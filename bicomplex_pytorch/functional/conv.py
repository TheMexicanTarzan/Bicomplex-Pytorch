"""
Functional activation functions for bicomplex neural networks.

All functions operate on bicomplex tensors in idempotent form (e1, e2).
"""

import torch
from typing import Optional, Literal, Union

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


def bicomplex_conv1d(
        input: tuple[torch.Tensor, torch.Tensor],
        weight: tuple[torch.Tensor, torch.Tensor],
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        stride: Union[int, tuple[int]] = 1,
        padding: Union[int, tuple[int], str] = 0,
        dilation: Union[int, tuple[int]] = 1,
        groups: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a 1D convolution to bicomplex input in idempotent form.

    In idempotent form, convolution is component-wise:
        output_e1 = conv1d(input_e1, weight_e1, bias_e1, ...)
        output_e2 = conv1d(input_e2, weight_e2, bias_e2, ...)

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, in_channels, length) for each component
        weight: Bicomplex weight tensor in idempotent form
                Shape: (out_channels, in_channels/groups, kernel_size) for each component
        bias: Optional bicomplex bias tensor in idempotent form
              Shape: (out_channels,) for each component
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides of the input (default: 0)
                 Can be int, tuple, or 'valid'/'same'
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections from input to output channels (default: 1)

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (batch, out_channels, output_length) for each component

    Raises:
        ValueError: If inputs are not valid bicomplex tensors in idempotent form

    Example:
        >>> input = (torch.randn(8, 16, 100), torch.randn(8, 16, 100))  # batch=8, channels=16, length=100
        >>> weight = (torch.randn(32, 16, 5), torch.randn(32, 16, 5))   # out=32, in=16, kernel=5
        >>> bias = (torch.randn(32), torch.randn(32))
        >>> output = bicomplex_conv1d(input, weight, bias, stride=1, padding=2)
        >>> output[0].shape, output[1].shape
        (torch.Size([8, 32, 100]), torch.Size([8, 32, 100]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    # Perform 1D convolution for e1 component
    output_e1 = torch.nn.functional.conv1d(
        input[0], weight[0], None, stride, padding, dilation, groups
    )

    # Perform 1D convolution for e2 component
    output_e2 = torch.nn.functional.conv1d(
        input[1], weight[1], None, stride, padding, dilation, groups
    )

    # Add bias if provided
    if bias is not None:
        # Bias shape: (out_channels,) -> need to reshape for broadcasting
        output_e1 = output_e1 + bias[0].view(1, -1, 1)
        output_e2 = output_e2 + bias[1].view(1, -1, 1)

    return (output_e1, output_e2)


def bicomplex_conv2d(
        input: tuple[torch.Tensor, torch.Tensor],
        weight: tuple[torch.Tensor, torch.Tensor],
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        stride: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int], str] = 0,
        dilation: Union[int, tuple[int, int]] = 1,
        groups: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a 2D convolution to bicomplex input in idempotent form.

    In idempotent form, convolution is component-wise:
        output_e1 = conv2d(input_e1, weight_e1, bias_e1, ...)
        output_e2 = conv2d(input_e2, weight_e2, bias_e2, ...)

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, in_channels, height, width) for each component
        weight: Bicomplex weight tensor in idempotent form
                Shape: (out_channels, in_channels/groups, kernel_h, kernel_w) for each component
        bias: Optional bicomplex bias tensor in idempotent form
              Shape: (out_channels,) for each component
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides of the input (default: 0)
                 Can be int, tuple, or 'valid'/'same'
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections from input to output channels (default: 1)

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (batch, out_channels, output_h, output_w) for each component

    Example:
        >>> input = (torch.randn(8, 3, 32, 32), torch.randn(8, 3, 32, 32))  # batch=8, channels=3, 32x32
        >>> weight = (torch.randn(16, 3, 3, 3), torch.randn(16, 3, 3, 3))   # out=16, in=3, 3x3 kernel
        >>> bias = (torch.randn(16), torch.randn(16))
        >>> output = bicomplex_conv2d(input, weight, bias, stride=1, padding=1)
        >>> output[0].shape, output[1].shape
        (torch.Size([8, 16, 32, 32]), torch.Size([8, 16, 32, 32]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    # Perform 2D convolution for e1 component
    output_e1 = torch.nn.functional.conv2d(
        input[0], weight[0], None, stride, padding, dilation, groups
    )

    # Perform 2D convolution for e2 component
    output_e2 = torch.nn.functional.conv2d(
        input[1], weight[1], None, stride, padding, dilation, groups
    )

    # Add bias if provided
    if bias is not None:
        # Bias shape: (out_channels,) -> need to reshape for broadcasting
        output_e1 = output_e1 + bias[0].view(1, -1, 1, 1)
        output_e2 = output_e2 + bias[1].view(1, -1, 1, 1)

    return (output_e1, output_e2)


def bicomplex_conv3d(
        input: tuple[torch.Tensor, torch.Tensor],
        weight: tuple[torch.Tensor, torch.Tensor],
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        stride: Union[int, tuple[int, int, int]] = 1,
        padding: Union[int, tuple[int, int, int], str] = 0,
        dilation: Union[int, tuple[int, int, int]] = 1,
        groups: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a 3D convolution to bicomplex input in idempotent form.

    In idempotent form, convolution is component-wise:
        output_e1 = conv3d(input_e1, weight_e1, bias_e1, ...)
        output_e2 = conv3d(input_e2, weight_e2, bias_e2, ...)

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, in_channels, depth, height, width) for each component
        weight: Bicomplex weight tensor in idempotent form
                Shape: (out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w)
                for each component
        bias: Optional bicomplex bias tensor in idempotent form
              Shape: (out_channels,) for each component
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides of the input (default: 0)
                 Can be int, tuple, or 'valid'/'same'
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections from input to output channels (default: 1)

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (batch, out_channels, output_d, output_h, output_w) for each component

    Example:
        >>> # batch=4, channels=8, 16x16x16 volume
        >>> input = (torch.randn(4, 8, 16, 16, 16), torch.randn(4, 8, 16, 16, 16))
        >>> # out=16, in=8, 3x3x3 kernel
        >>> weight = (torch.randn(16, 8, 3, 3, 3), torch.randn(16, 8, 3, 3, 3))
        >>> bias = (torch.randn(16), torch.randn(16))
        >>> output = bicomplex_conv3d(input, weight, bias, stride=1, padding=1)
        >>> output[0].shape, output[1].shape
        (torch.Size([4, 16, 16, 16, 16]), torch.Size([4, 16, 16, 16, 16]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    # Perform 3D convolution for e1 component
    output_e1 = torch.nn.functional.conv3d(
        input[0], weight[0], None, stride, padding, dilation, groups
    )

    # Perform 3D convolution for e2 component
    output_e2 = torch.nn.functional.conv3d(
        input[1], weight[1], None, stride, padding, dilation, groups
    )

    # Add bias if provided
    if bias is not None:
        # Bias shape: (out_channels,) -> need to reshape for broadcasting
        output_e1 = output_e1 + bias[0].view(1, -1, 1, 1, 1)
        output_e2 = output_e2 + bias[1].view(1, -1, 1, 1, 1)

    return (output_e1, output_e2)


def bicomplex_conv_transpose1d(
        input: tuple[torch.Tensor, torch.Tensor],
        weight: tuple[torch.Tensor, torch.Tensor],
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        stride: Union[int, tuple[int]] = 1,
        padding: Union[int, tuple[int]] = 0,
        output_padding: Union[int, tuple[int]] = 0,
        groups: int = 1,
        dilation: Union[int, tuple[int]] = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a 1D transposed convolution (deconvolution) to bicomplex input.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, in_channels, length) for each component
        weight: Bicomplex weight tensor in idempotent form
                Shape: (in_channels, out_channels/groups, kernel_size) for each component
        bias: Optional bicomplex bias tensor in idempotent form
              Shape: (out_channels,) for each component
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides of the input (default: 0)
        output_padding: Additional size added to output shape (default: 0)
        groups: Number of blocked connections (default: 1)
        dilation: Spacing between kernel elements (default: 1)

    Returns:
        Bicomplex output tensor in idempotent form
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.conv_transpose1d(
        input[0], weight[0], None, stride, padding, output_padding, groups, dilation
    )

    output_e2 = torch.nn.functional.conv_transpose1d(
        input[1], weight[1], None, stride, padding, output_padding, groups, dilation
    )

    if bias is not None:
        output_e1 = output_e1 + bias[0].view(1, -1, 1)
        output_e2 = output_e2 + bias[1].view(1, -1, 1)

    return (output_e1, output_e2)


def bicomplex_conv_transpose2d(
        input: tuple[torch.Tensor, torch.Tensor],
        weight: tuple[torch.Tensor, torch.Tensor],
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        stride: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int]] = 0,
        output_padding: Union[int, tuple[int, int]] = 0,
        groups: int = 1,
        dilation: Union[int, tuple[int, int]] = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a 2D transposed convolution (deconvolution) to bicomplex input.

    Useful for upsampling in decoder networks, GANs, and autoencoders.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, in_channels, height, width) for each component
        weight: Bicomplex weight tensor in idempotent form
                Shape: (in_channels, out_channels/groups, kernel_h, kernel_w) for each component
        bias: Optional bicomplex bias tensor in idempotent form
              Shape: (out_channels,) for each component
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides of the input (default: 0)
        output_padding: Additional size added to output shape (default: 0)
        groups: Number of blocked connections (default: 1)
        dilation: Spacing between kernel elements (default: 1)

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (batch, out_channels, output_h, output_w) for each component

    Example:
        >>> # Upsampling from 8x8 to 16x16
        >>> input = (torch.randn(4, 32, 8, 8), torch.randn(4, 32, 8, 8))
        >>> weight = (torch.randn(32, 16, 4, 4), torch.randn(32, 16, 4, 4))
        >>> output = bicomplex_conv_transpose2d(input, weight, stride=2, padding=1)
        >>> output[0].shape, output[1].shape
        (torch.Size([4, 16, 16, 16]), torch.Size([4, 16, 16, 16]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.conv_transpose2d(
        input[0], weight[0], None, stride, padding, output_padding, groups, dilation
    )

    output_e2 = torch.nn.functional.conv_transpose2d(
        input[1], weight[1], None, stride, padding, output_padding, groups, dilation
    )

    if bias is not None:
        output_e1 = output_e1 + bias[0].view(1, -1, 1, 1)
        output_e2 = output_e2 + bias[1].view(1, -1, 1, 1)

    return (output_e1, output_e2)


def bicomplex_conv_transpose3d(
        input: tuple[torch.Tensor, torch.Tensor],
        weight: tuple[torch.Tensor, torch.Tensor],
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        stride: Union[int, tuple[int, int, int]] = 1,
        padding: Union[int, tuple[int, int, int]] = 0,
        output_padding: Union[int, tuple[int, int, int]] = 0,
        groups: int = 1,
        dilation: Union[int, tuple[int, int, int]] = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a 3D transposed convolution (deconvolution) to bicomplex input.

    Useful for 3D upsampling in medical imaging, video processing, and volumetric data.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, in_channels, depth, height, width) for each component
        weight: Bicomplex weight tensor in idempotent form
                Shape: (in_channels, out_channels/groups, kernel_d, kernel_h, kernel_w)
                for each component
        bias: Optional bicomplex bias tensor in idempotent form
              Shape: (out_channels,) for each component
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides of the input (default: 0)
        output_padding: Additional size added to output shape (default: 0)
        groups: Number of blocked connections (default: 1)
        dilation: Spacing between kernel elements (default: 1)

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (batch, out_channels, output_d, output_h, output_w) for each component
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.conv_transpose3d(
        input[0], weight[0], None, stride, padding, output_padding, groups, dilation
    )

    output_e2 = torch.nn.functional.conv_transpose3d(
        input[1], weight[1], None, stride, padding, output_padding, groups, dilation
    )

    if bias is not None:
        output_e1 = output_e1 + bias[0].view(1, -1, 1, 1, 1)
        output_e2 = output_e2 + bias[1].view(1, -1, 1, 1, 1)

    return (output_e1, output_e2)


def bicomplex_unfold(
        input: tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, tuple[int, int]],
        dilation: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int]] = 0,
        stride: Union[int, tuple[int, int]] = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts sliding local blocks from bicomplex input tensor.

    Useful for implementing custom convolution operations or patch-based processing.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, height, width) for each component
        kernel_size: Size of the sliding blocks
        dilation: Spacing between kernel elements (default: 1)
        padding: Zero-padding added to input (default: 0)
        stride: Stride of the sliding blocks (default: 1)

    Returns:
        Bicomplex output tensor with unfolded blocks
        Shape: (batch, channels * ∏kernel_size, num_blocks) for each component

    Example:
        >>> input = (torch.randn(2, 3, 10, 10), torch.randn(2, 3, 10, 10))
        >>> output = bicomplex_unfold(input, kernel_size=3, padding=1, stride=1)
        >>> output[0].shape, output[1].shape
        (torch.Size([2, 27, 100]), torch.Size([2, 27, 100]))  # 3*3*3=27, 10*10=100
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.unfold(
        input[0], kernel_size, dilation, padding, stride
    )

    output_e2 = torch.nn.functional.unfold(
        input[1], kernel_size, dilation, padding, stride
    )

    return (output_e1, output_e2)


def bicomplex_fold(
        input: tuple[torch.Tensor, torch.Tensor],
        output_size: tuple[int, int],
        kernel_size: Union[int, tuple[int, int]],
        dilation: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int]] = 0,
        stride: Union[int, tuple[int, int]] = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combines sliding local blocks into a large containing tensor.

    Inverse operation of unfold.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels * ∏kernel_size, num_blocks) for each component
        output_size: Shape of the output (height, width)
        kernel_size: Size of the sliding blocks
        dilation: Spacing between kernel elements (default: 1)
        padding: Zero-padding (default: 0)
        stride: Stride of the sliding blocks (default: 1)

    Returns:
        Bicomplex output tensor
        Shape: (batch, channels, height, width) for each component
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.fold(
        input[0], output_size, kernel_size, dilation, padding, stride
    )

    output_e2 = torch.nn.functional.fold(
        input[1], output_size, kernel_size, dilation, padding, stride
    )

    return (output_e1, output_e2)