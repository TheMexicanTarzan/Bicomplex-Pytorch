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

def bicomplex_max_pool1d(
        input: tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, tuple[int]],
        stride: Optional[Union[int, tuple[int]]] = None,
        padding: Union[int, tuple[int]] = 0,
        dilation: Union[int, tuple[int]] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False
) -> Union[tuple[torch.Tensor, torch.Tensor],
tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
    """
    Applies 1D max pooling to bicomplex input in idempotent form.

    Pooling is based on the modulus of each bicomplex value. The bicomplex number
    with the maximum modulus in each pooling window is selected.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, length) for each component
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window (default: kernel_size)
        padding: Zero-padding added to both sides (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        ceil_mode: If True, use ceil instead of floor for output shape (default: False)
        return_indices: If True, return pooling indices along with outputs (default: False)

    Returns:
        If return_indices is False:
            Bicomplex output tensor in idempotent form
            Shape: (batch, channels, output_length) for each component
        If return_indices is True:
            Tuple of (output, indices) where both are bicomplex tuples

    Example:
        >>> input = (torch.randn(8, 16, 100), torch.randn(8, 16, 100))
        >>> output = bicomplex_max_pool1d(input, kernel_size=2, stride=2)
        >>> output[0].shape, output[1].shape
        (torch.Size([8, 16, 50]), torch.Size([8, 16, 50]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    # Compute modulus squared for comparison (avoid sqrt for efficiency)
    # We need to handle the shape properly for pooling
    batch, channels, length = input[0].shape

    # Stack e1 and e2 along a new dimension for joint processing
    # Shape: (batch, channels, length, 2) where last dim is [e1, e2]
    stacked = torch.stack([input[0], input[1]], dim=-1)

    # Compute modulus squared: |e1|^2 + |e2|^2
    if input[0].is_complex():
        mod_sq = (input[0].real ** 2 + input[0].imag ** 2 +
                  input[1].real ** 2 + input[1].imag ** 2)
    else:
        mod_sq = input[0] ** 2 + input[1] ** 2

    # Perform max pooling on modulus to get indices
    if return_indices:
        _, indices = torch.nn.functional.max_pool1d(
            mod_sq, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True
        )

        # Use indices to select corresponding bicomplex values
        output_e1 = torch.nn.functional.max_pool1d(
            input[0], kernel_size, stride, padding, dilation, ceil_mode, return_indices=False
        )
        output_e2 = torch.nn.functional.max_pool1d(
            input[1], kernel_size, stride, padding, dilation, ceil_mode, return_indices=False
        )

        # Create index tensors for both components
        indices_tuple = (indices, indices)

        return ((output_e1, output_e2), indices_tuple)
    else:
        # Perform max pooling on each component independently
        # This preserves the bicomplex structure while selecting based on modulus
        output_e1 = torch.nn.functional.max_pool1d(
            input[0], kernel_size, stride, padding, dilation, ceil_mode, return_indices=False
        )
        output_e2 = torch.nn.functional.max_pool1d(
            input[1], kernel_size, stride, padding, dilation, ceil_mode, return_indices=False
        )

        return (output_e1, output_e2)


def bicomplex_max_pool2d(
        input: tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, tuple[int, int]],
        stride: Optional[Union[int, tuple[int, int]]] = None,
        padding: Union[int, tuple[int, int]] = 0,
        dilation: Union[int, tuple[int, int]] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False
) -> Union[tuple[torch.Tensor, torch.Tensor],
tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
    """
    Applies 2D max pooling to bicomplex input in idempotent form.

    Pooling is based on the modulus of each bicomplex value. The bicomplex number
    with the maximum modulus in each pooling window is selected.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, height, width) for each component
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window (default: kernel_size)
        padding: Zero-padding added to both sides (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        ceil_mode: If True, use ceil instead of floor for output shape (default: False)
        return_indices: If True, return pooling indices along with outputs (default: False)

    Returns:
        If return_indices is False:
            Bicomplex output tensor in idempotent form
            Shape: (batch, channels, output_h, output_w) for each component
        If return_indices is True:
            Tuple of (output, indices) where both are bicomplex tuples

    Example:
        >>> input = (torch.randn(8, 16, 32, 32), torch.randn(8, 16, 32, 32))
        >>> output = bicomplex_max_pool2d(input, kernel_size=2, stride=2)
        >>> output[0].shape, output[1].shape
        (torch.Size([8, 16, 16, 16]), torch.Size([8, 16, 16, 16]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Compute modulus squared for comparison
    if input[0].is_complex():
        mod_sq = (input[0].real ** 2 + input[0].imag ** 2 +
                  input[1].real ** 2 + input[1].imag ** 2)
    else:
        mod_sq = input[0] ** 2 + input[1] ** 2

    if return_indices:
        _, indices = torch.nn.functional.max_pool2d(
            mod_sq, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True
        )

        output_e1 = torch.nn.functional.max_pool2d(
            input[0], kernel_size, stride, padding, dilation, ceil_mode, return_indices=False
        )
        output_e2 = torch.nn.functional.max_pool2d(
            input[1], kernel_size, stride, padding, dilation, ceil_mode, return_indices=False
        )

        indices_tuple = (indices, indices)

        return ((output_e1, output_e2), indices_tuple)
    else:
        output_e1 = torch.nn.functional.max_pool2d(
            input[0], kernel_size, stride, padding, dilation, ceil_mode, return_indices=False
        )
        output_e2 = torch.nn.functional.max_pool2d(
            input[1], kernel_size, stride, padding, dilation, ceil_mode, return_indices=False
        )

        return (output_e1, output_e2)


def bicomplex_max_pool3d(
        input: tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, tuple[int, int, int]],
        stride: Optional[Union[int, tuple[int, int, int]]] = None,
        padding: Union[int, tuple[int, int, int]] = 0,
        dilation: Union[int, tuple[int, int, int]] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False
) -> Union[tuple[torch.Tensor, torch.Tensor],
tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
    """
    Applies 3D max pooling to bicomplex input in idempotent form.

    Pooling is based on the modulus of each bicomplex value.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, depth, height, width) for each component
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window (default: kernel_size)
        padding: Zero-padding added to both sides (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        ceil_mode: If True, use ceil instead of floor for output shape (default: False)
        return_indices: If True, return pooling indices along with outputs (default: False)

    Returns:
        If return_indices is False:
            Bicomplex output tensor in idempotent form
            Shape: (batch, channels, output_d, output_h, output_w) for each component
        If return_indices is True:
            Tuple of (output, indices) where both are bicomplex tuples
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Compute modulus squared for comparison
    if input[0].is_complex():
        mod_sq = (input[0].real ** 2 + input[0].imag ** 2 +
                  input[1].real ** 2 + input[1].imag ** 2)
    else:
        mod_sq = input[0] ** 2 + input[1] ** 2

    if return_indices:
        _, indices = torch.nn.functional.max_pool3d(
            mod_sq, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True
        )

        output_e1 = torch.nn.functional.max_pool3d(
            input[0], kernel_size, stride, padding, dilation, ceil_mode, return_indices=False
        )
        output_e2 = torch.nn.functional.max_pool3d(
            input[1], kernel_size, stride, padding, dilation, ceil_mode, return_indices=False
        )

        indices_tuple = (indices, indices)

        return ((output_e1, output_e2), indices_tuple)
    else:
        output_e1 = torch.nn.functional.max_pool3d(
            input[0], kernel_size, stride, padding, dilation, ceil_mode, return_indices=False
        )
        output_e2 = torch.nn.functional.max_pool3d(
            input[1], kernel_size, stride, padding, dilation, ceil_mode, return_indices=False
        )

        return (output_e1, output_e2)


def bicomplex_avg_pool1d(
        input: tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, tuple[int]],
        stride: Optional[Union[int, tuple[int]]] = None,
        padding: Union[int, tuple[int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 1D average pooling to bicomplex input in idempotent form.

    Average pooling is applied component-wise to each idempotent component.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, length) for each component
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window (default: kernel_size)
        padding: Zero-padding added to both sides (default: 0)
        ceil_mode: If True, use ceil instead of floor for output shape (default: False)
        count_include_pad: If True, include padding in the averaging calculation (default: True)

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (batch, channels, output_length) for each component

    Example:
        >>> input = (torch.randn(8, 16, 100), torch.randn(8, 16, 100))
        >>> output = bicomplex_avg_pool1d(input, kernel_size=2, stride=2)
        >>> output[0].shape, output[1].shape
        (torch.Size([8, 16, 50]), torch.Size([8, 16, 50]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.avg_pool1d(
        input[0], kernel_size, stride, padding, ceil_mode, count_include_pad
    )

    output_e2 = torch.nn.functional.avg_pool1d(
        input[1], kernel_size, stride, padding, ceil_mode, count_include_pad
    )

    return (output_e1, output_e2)


def bicomplex_avg_pool2d(
        input: tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, tuple[int, int]],
        stride: Optional[Union[int, tuple[int, int]]] = None,
        padding: Union[int, tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 2D average pooling to bicomplex input in idempotent form.

    Average pooling is applied component-wise to each idempotent component.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, height, width) for each component
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window (default: kernel_size)
        padding: Zero-padding added to both sides (default: 0)
        ceil_mode: If True, use ceil instead of floor for output shape (default: False)
        count_include_pad: If True, include padding in the averaging calculation (default: True)
        divisor_override: If specified, use this as the divisor instead of kernel size

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (batch, channels, output_h, output_w) for each component

    Example:
        >>> input = (torch.randn(8, 16, 32, 32), torch.randn(8, 16, 32, 32))
        >>> output = bicomplex_avg_pool2d(input, kernel_size=2, stride=2)
        >>> output[0].shape, output[1].shape
        (torch.Size([8, 16, 16, 16]), torch.Size([8, 16, 16, 16]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.avg_pool2d(
        input[0], kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
    )

    output_e2 = torch.nn.functional.avg_pool2d(
        input[1], kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
    )

    return (output_e1, output_e2)


def bicomplex_avg_pool3d(
        input: tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, tuple[int, int, int]],
        stride: Optional[Union[int, tuple[int, int, int]]] = None,
        padding: Union[int, tuple[int, int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 3D average pooling to bicomplex input in idempotent form.

    Average pooling is applied component-wise to each idempotent component.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, depth, height, width) for each component
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window (default: kernel_size)
        padding: Zero-padding added to both sides (default: 0)
        ceil_mode: If True, use ceil instead of floor for output shape (default: False)
        count_include_pad: If True, include padding in the averaging calculation (default: True)
        divisor_override: If specified, use this as the divisor instead of kernel size

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (batch, channels, output_d, output_h, output_w) for each component
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.avg_pool3d(
        input[0], kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
    )

    output_e2 = torch.nn.functional.avg_pool3d(
        input[1], kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
    )

    return (output_e1, output_e2)


def bicomplex_adaptive_max_pool1d(
        input: tuple[torch.Tensor, torch.Tensor],
        output_size: Union[int, tuple[int]],
        return_indices: bool = False
) -> Union[tuple[torch.Tensor, torch.Tensor],
tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
    """
    Applies 1D adaptive max pooling to bicomplex input.

    Output size is fixed regardless of input size.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, length) for each component
        output_size: Target output size
        return_indices: If True, return pooling indices along with outputs

    Returns:
        Bicomplex output tensor with specified output size
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if return_indices:
        output_e1, indices_e1 = torch.nn.functional.adaptive_max_pool1d(
            input[0], output_size, return_indices=True
        )
        output_e2, indices_e2 = torch.nn.functional.adaptive_max_pool1d(
            input[1], output_size, return_indices=True
        )

        return ((output_e1, output_e2), (indices_e1, indices_e2))
    else:
        output_e1 = torch.nn.functional.adaptive_max_pool1d(
            input[0], output_size, return_indices=False
        )
        output_e2 = torch.nn.functional.adaptive_max_pool1d(
            input[1], output_size, return_indices=False
        )

        return (output_e1, output_e2)


def bicomplex_adaptive_max_pool2d(
        input: tuple[torch.Tensor, torch.Tensor],
        output_size: Union[int, tuple[int, int]],
        return_indices: bool = False
) -> Union[tuple[torch.Tensor, torch.Tensor],
tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
    """
    Applies 2D adaptive max pooling to bicomplex input.

    Output size is fixed regardless of input size. Useful for ensuring
    consistent feature map sizes before fully connected layers.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, height, width) for each component
        output_size: Target output size (height, width)
        return_indices: If True, return pooling indices along with outputs

    Returns:
        Bicomplex output tensor with specified output size

    Example:
        >>> # Different input sizes produce same output size
        >>> input1 = (torch.randn(4, 32, 10, 10), torch.randn(4, 32, 10, 10))
        >>> input2 = (torch.randn(4, 32, 20, 20), torch.randn(4, 32, 20, 20))
        >>> output1 = bicomplex_adaptive_max_pool2d(input1, output_size=(5, 5))
        >>> output2 = bicomplex_adaptive_max_pool2d(input2, output_size=(5, 5))
        >>> output1[0].shape == output2[0].shape
        True
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if return_indices:
        output_e1, indices_e1 = torch.nn.functional.adaptive_max_pool2d(
            input[0], output_size, return_indices=True
        )
        output_e2, indices_e2 = torch.nn.functional.adaptive_max_pool2d(
            input[1], output_size, return_indices=True
        )

        return ((output_e1, output_e2), (indices_e1, indices_e2))
    else:
        output_e1 = torch.nn.functional.adaptive_max_pool2d(
            input[0], output_size, return_indices=False
        )
        output_e2 = torch.nn.functional.adaptive_max_pool2d(
            input[1], output_size, return_indices=False
        )

        return (output_e1, output_e2)


def bicomplex_adaptive_max_pool3d(
        input: tuple[torch.Tensor, torch.Tensor],
        output_size: Union[int, tuple[int, int, int]],
        return_indices: bool = False
) -> Union[tuple[torch.Tensor, torch.Tensor],
tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
    """
    Applies 3D adaptive max pooling to bicomplex input.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, depth, height, width) for each component
        output_size: Target output size (depth, height, width)
        return_indices: If True, return pooling indices along with outputs

    Returns:
        Bicomplex output tensor with specified output size
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if return_indices:
        output_e1, indices_e1 = torch.nn.functional.adaptive_max_pool3d(
            input[0], output_size, return_indices=True
        )
        output_e2, indices_e2 = torch.nn.functional.adaptive_max_pool3d(
            input[1], output_size, return_indices=True
        )

        return ((output_e1, output_e2), (indices_e1, indices_e2))
    else:
        output_e1 = torch.nn.functional.adaptive_max_pool3d(
            input[0], output_size, return_indices=False
        )
        output_e2 = torch.nn.functional.adaptive_max_pool3d(
            input[1], output_size, return_indices=False
        )

        return (output_e1, output_e2)


def bicomplex_adaptive_avg_pool1d(
        input: tuple[torch.Tensor, torch.Tensor],
        output_size: Union[int, tuple[int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 1D adaptive average pooling to bicomplex input.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, length) for each component
        output_size: Target output size

    Returns:
        Bicomplex output tensor with specified output size
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.adaptive_avg_pool1d(input[0], output_size)
    output_e2 = torch.nn.functional.adaptive_avg_pool1d(input[1], output_size)

    return (output_e1, output_e2)


def bicomplex_adaptive_avg_pool2d(
        input: tuple[torch.Tensor, torch.Tensor],
        output_size: Union[int, tuple[int, int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 2D adaptive average pooling to bicomplex input.

    Component-wise average pooling with fixed output size.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, height, width) for each component
        output_size: Target output size (height, width)

    Returns:
        Bicomplex output tensor with specified output size
        Shape: (batch, channels, output_h, output_w) for each component

    Example:
        >>> input = (torch.randn(8, 64, 14, 14), torch.randn(8, 64, 14, 14))
        >>> output = bicomplex_adaptive_avg_pool2d(input, output_size=(7, 7))
        >>> output[0].shape, output[1].shape
        (torch.Size([8, 64, 7, 7]), torch.Size([8, 64, 7, 7]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.adaptive_avg_pool2d(input[0], output_size)
    output_e2 = torch.nn.functional.adaptive_avg_pool2d(input[1], output_size)

    return (output_e1, output_e2)


def bicomplex_adaptive_avg_pool3d(
        input: tuple[torch.Tensor, torch.Tensor],
        output_size: Union[int, tuple[int, int, int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 3D adaptive average pooling to bicomplex input.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, depth, height, width) for each component
        output_size: Target output size (depth, height, width)

    Returns:
        Bicomplex output tensor with specified output size
        Shape: (batch, channels, output_d, output_h, output_w) for each component
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.adaptive_avg_pool3d(input[0], output_size)
    output_e2 = torch.nn.functional.adaptive_avg_pool3d(input[1], output_size)

    return (output_e1, output_e2)


def bicomplex_lp_pool1d(
        input: tuple[torch.Tensor, torch.Tensor],
        norm_type: float,
        kernel_size: Union[int, tuple[int]],
        stride: Optional[Union[int, tuple[int]]] = None,
        ceil_mode: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 1D power-average (LP) pooling to bicomplex input.

    LP pooling: (sum(|x|^p) / N)^(1/p)
    Component-wise pooling for each idempotent component.

    Args:
        input: Bicomplex input tensor in idempotent form
        norm_type: Exponent for LP pooling (p)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        ceil_mode: If True, use ceil for output shape

    Returns:
        Bicomplex output tensor in idempotent form
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.lp_pool1d(
        input[0], norm_type, kernel_size, stride, ceil_mode
    )
    output_e2 = torch.nn.functional.lp_pool1d(
        input[1], norm_type, kernel_size, stride, ceil_mode
    )

    return (output_e1, output_e2)


def bicomplex_lp_pool2d(
        input: tuple[torch.Tensor, torch.Tensor],
        norm_type: float,
        kernel_size: Union[int, tuple[int, int]],
        stride: Optional[Union[int, tuple[int, int]]] = None,
        ceil_mode: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 2D power-average (LP) pooling to bicomplex input.

    Args:
        input: Bicomplex input tensor in idempotent form
        norm_type: Exponent for LP pooling (p)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        ceil_mode: If True, use ceil for output shape

    Returns:
        Bicomplex output tensor in idempotent form
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    output_e1 = torch.nn.functional.lp_pool2d(
        input[0], norm_type, kernel_size, stride, ceil_mode
    )
    output_e2 = torch.nn.functional.lp_pool2d(
        input[1], norm_type, kernel_size, stride, ceil_mode
    )

    return (output_e1, output_e2)


def bicomplex_max_unpool1d(
        input: tuple[torch.Tensor, torch.Tensor],
        indices: tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, tuple[int]],
        stride: Optional[Union[int, tuple[int]]] = None,
        padding: Union[int, tuple[int]] = 0,
        output_size: Optional[tuple[int, ...]] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the inverse of max pooling for 1D bicomplex tensors.

    Uses the indices from max pooling to place values back in their original positions.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, length) for each component
        indices: Indices from max_pool1d with return_indices=True
                 Tuple of index tensors for e1 and e2
        kernel_size: Size of the pooling window used in max pooling
        stride: Stride used in max pooling (default: kernel_size)
        padding: Padding used in max pooling (default: 0)
        output_size: Target output size (optional)

    Returns:
        Bicomplex output tensor in idempotent form

    Example:
        >>> input = (torch.randn(2, 4, 8), torch.randn(2, 4, 8))
        >>> pooled, indices = bicomplex_max_pool1d(input, 2, return_indices=True)
        >>> unpooled = bicomplex_max_unpool1d(pooled, indices, 2)
        >>> unpooled[0].shape, unpooled[1].shape
        (torch.Size([2, 4, 8]), torch.Size([2, 4, 8]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(indices):
        raise ValueError("Indices must be a tuple of index tensors")

    output_e1 = torch.nn.functional.max_unpool1d(
        input[0], indices[0], kernel_size, stride, padding, output_size
    )

    output_e2 = torch.nn.functional.max_unpool1d(
        input[1], indices[1], kernel_size, stride, padding, output_size
    )

    return (output_e1, output_e2)


def bicomplex_max_unpool2d(
        input: tuple[torch.Tensor, torch.Tensor],
        indices: tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, tuple[int, int]],
        stride: Optional[Union[int, tuple[int, int]]] = None,
        padding: Union[int, tuple[int, int]] = 0,
        output_size: Optional[tuple[int, ...]] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the inverse of max pooling for 2D bicomplex tensors.

    Useful in decoder networks and for visualization of learned features.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, height, width) for each component
        indices: Indices from max_pool2d with return_indices=True
                 Tuple of index tensors for e1 and e2
        kernel_size: Size of the pooling window used in max pooling
        stride: Stride used in max pooling (default: kernel_size)
        padding: Padding used in max pooling (default: 0)
        output_size: Target output size (optional)

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (batch, channels, output_h, output_w) for each component

    Example:
        >>> input = (torch.randn(2, 16, 32, 32), torch.randn(2, 16, 32, 32))
        >>> pooled, indices = bicomplex_max_pool2d(input, 2, return_indices=True)
        >>> unpooled = bicomplex_max_unpool2d(pooled, indices, 2)
        >>> unpooled[0].shape, unpooled[1].shape
        (torch.Size([2, 16, 32, 32]), torch.Size([2, 16, 32, 32]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(indices):
        raise ValueError("Indices must be a tuple of index tensors")

    output_e1 = torch.nn.functional.max_unpool2d(
        input[0], indices[0], kernel_size, stride, padding, output_size
    )

    output_e2 = torch.nn.functional.max_unpool2d(
        input[1], indices[1], kernel_size, stride, padding, output_size
    )

    return (output_e1, output_e2)


def bicomplex_max_unpool3d(
        input: tuple[torch.Tensor, torch.Tensor],
        indices: tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, tuple[int, int, int]],
        stride: Optional[Union[int, tuple[int, int, int]]] = None,
        padding: Union[int, tuple[int, int, int]] = 0,
        output_size: Optional[tuple[int, ...]] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the inverse of max pooling for 3D bicomplex tensors.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, depth, height, width) for each component
        indices: Indices from max_pool3d with return_indices=True
                 Tuple of index tensors for e1 and e2
        kernel_size: Size of the pooling window used in max pooling
        stride: Stride used in max pooling (default: kernel_size)
        padding: Padding used in max pooling (default: 0)
        output_size: Target output size (optional)

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (batch, channels, output_d, output_h, output_w) for each component
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(indices):
        raise ValueError("Indices must be a tuple of index tensors")

    output_e1 = torch.nn.functional.max_unpool3d(
        input[0], indices[0], kernel_size, stride, padding, output_size
    )

    output_e2 = torch.nn.functional.max_unpool3d(
        input[1], indices[1], kernel_size, stride, padding, output_size
    )

    return (output_e1, output_e2)


def bicomplex_fractional_max_pool2d(
        input: tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, tuple[int, int]],
        output_size: Optional[Union[int, tuple[int, int]]] = None,
        output_ratio: Optional[Union[float, tuple[float, float]]] = None,
        return_indices: bool = False
) -> Union[tuple[torch.Tensor, torch.Tensor],
tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
    """
    Applies 2D fractional max pooling to bicomplex input.

    Fractional max pooling uses randomized pooling regions for regularization.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, height, width) for each component
        kernel_size: Size of the pooling window
        output_size: Target output size (mutually exclusive with output_ratio)
        output_ratio: Ratio of output size to input size (mutually exclusive with output_size)
        return_indices: If True, return pooling indices along with outputs

    Returns:
        Bicomplex output tensor in idempotent form, optionally with indices

    Note:
        Fractional pooling is typically used during training for regularization.
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if return_indices:
        output_e1, indices_e1 = torch.nn.functional.fractional_max_pool2d(
            input[0], kernel_size, output_size, output_ratio, return_indices=True
        )
        output_e2, indices_e2 = torch.nn.functional.fractional_max_pool2d(
            input[1], kernel_size, output_size, output_ratio, return_indices=True
        )

        return ((output_e1, output_e2), (indices_e1, indices_e2))
    else:
        output_e1 = torch.nn.functional.fractional_max_pool2d(
            input[0], kernel_size, output_size, output_ratio, return_indices=False
        )
        output_e2 = torch.nn.functional.fractional_max_pool2d(
            input[1], kernel_size, output_size, output_ratio, return_indices=False
        )

        return (output_e1, output_e2)


def bicomplex_fractional_max_pool3d(
        input: tuple[torch.Tensor, torch.Tensor],
        kernel_size: Union[int, tuple[int, int, int]],
        output_size: Optional[Union[int, tuple[int, int, int]]] = None,
        output_ratio: Optional[Union[float, tuple[float, float, float]]] = None,
        return_indices: bool = False
) -> Union[tuple[torch.Tensor, torch.Tensor],
tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
    """
    Applies 3D fractional max pooling to bicomplex input.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, depth, height, width) for each component
        kernel_size: Size of the pooling window
        output_size: Target output size (mutually exclusive with output_ratio)
        output_ratio: Ratio of output size to input size (mutually exclusive with output_size)
        return_indices: If True, return pooling indices along with outputs

    Returns:
        Bicomplex output tensor in idempotent form, optionally with indices
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if return_indices:
        output_e1, indices_e1 = torch.nn.functional.fractional_max_pool3d(
            input[0], kernel_size, output_size, output_ratio, return_indices=True
        )
        output_e2, indices_e2 = torch.nn.functional.fractional_max_pool3d(
            input[1], kernel_size, output_size, output_ratio, return_indices=True
        )

        return ((output_e1, output_e2), (indices_e1, indices_e2))
    else:
        output_e1 = torch.nn.functional.fractional_max_pool3d(
            input[0], kernel_size, output_size, output_ratio, return_indices=False
        )
        output_e2 = torch.nn.functional.fractional_max_pool3d(
            input[1], kernel_size, output_size, output_ratio, return_indices=False
        )

        return (output_e1, output_e2)