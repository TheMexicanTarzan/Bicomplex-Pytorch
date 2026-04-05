"""
Complex and bicomplex convolution layers.

This module implements 2D and 3D convolution for complex- and bicomplex-valued
neural networks.

================================================================================
COMPLEX CONVOLUTION
================================================================================

ComplexConv2d and ComplexConv3d perform convolution on complex-valued tensors
using PyTorch's native complex tensor support. The weight and bias are complex
parameters, and the convolution is computed via a single call to the real
F.conv{2d,3d} backend (PyTorch handles the complex arithmetic internally).

Channel layout:
    - Input:  (N, C_in,  *spatial) complex tensor
    - Output: (N, C_out, *spatial) complex tensor

The coupled-bias strategy from the linear module is reused here: real bias
pairs (b_r, b_i) are combined as (b_r - b_i) + j*(b_r + b_i) so that each
underlying real parameter contributes to both real and imaginary output
channels, producing coupled gradients.

================================================================================
BICOMPLEX CONVOLUTION
================================================================================

BiComplexConv2d and BiComplexConv3d decompose bicomplex input into two
independent complex convolutions via the idempotent representation:

    (z1', z2') = (conv1(z1), conv2(z2))

This mirrors the approach used in BiComplexLinear.

Channel layout (idempotent form):
    - Input:  tuple of (N, C_in, *spatial) complex tensors
    - Output: tuple of (N, C_out, *spatial) complex tensors

================================================================================
WEIGHT CONFIGURATIONS (bicomplex only)
================================================================================

1. SHARED WEIGHTS (shared_weights=True):
   - Same convolution kernel for both idempotent components
   - Parameters: 1 complex conv kernel

2. INDEPENDENT WEIGHTS (shared_weights=False, default):
   - Different kernels for each component
   - Parameters: 2 complex conv kernels

================================================================================
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Union


from ...core.representations import to_idempotent, from_idempotent, is_idempotent, is_bicomplex


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _coupled_bias(b_r, b_i):
    """Build a complex bias from real pairs with coupled gradient structure."""
    if b_r is None:
        return None
    return torch.complex(b_r - b_i, b_r + b_i)


def _complex_kaiming_uniform_conv_(weight):
    """Initialize a complex conv weight with Kaiming uniform.

    Uses fan_in computed from the kernel shape, matching PyTorch Conv defaults.
    """
    # weight shape: (out_channels, in_channels, *kernel_size)
    fan_in = weight[0].numel()  # in_channels * prod(kernel_size)
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        real = torch.empty_like(weight.real).uniform_(-bound, bound)
        imag = torch.empty_like(weight.imag).uniform_(-bound, bound)
        weight.copy_(torch.complex(real, imag))


def _init_real_bias(bias, fan_in):
    """Initialize a real bias matching PyTorch conv defaults."""
    bound = 1.0 / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)


# ============================================================================
# COMPLEX CONVOLUTION LAYERS
# ============================================================================

class ComplexConv2d(nn.Module):
    """
    Complex-valued 2D convolution layer.

    Applies a 2D convolution over a complex-valued input signal.
    Uses complex weight parameters with real bias pairs for coupled
    gradient structure.

    Args:
        in_channels: Number of channels in the complex input
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 1
        padding: Zero-padding added to both sides of the input. Default: 0
        dilation: Spacing between kernel elements. Default: 1
        groups: Number of blocked connections from input to output. Default: 1
        bias: If True, adds a learnable bias. Default: True

    Shape:
        - Input: (N, C_in, H, W) complex tensor
        - Output: (N, C_out, H_out, W_out) complex tensor
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int]] = 0,
        dilation: Union[int, tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *self.kernel_size, dtype=torch.cfloat))

        if bias:
            self.bias_r = nn.Parameter(torch.empty(out_channels))
            self.bias_i = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_i', None)

        self.reset_parameters()

    def reset_parameters(self):
        _complex_kaiming_uniform_conv_(self.weight)
        if self.bias_r is not None:
            fan_in = self.weight[0].numel()
            _init_real_bias(self.bias_r, fan_in)
            _init_real_bias(self.bias_i, fan_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.weight, _coupled_bias(self.bias_r, self.bias_i),
            self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, dilation={self.dilation}, '
                f'groups={self.groups}, bias={self.bias_r is not None}')


class ComplexConv3d(nn.Module):
    """
    Complex-valued 3D convolution layer.

    Applies a 3D convolution over a complex-valued input signal.
    Uses complex weight parameters with real bias pairs for coupled
    gradient structure.

    Args:
        in_channels: Number of channels in the complex input
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 1
        padding: Zero-padding added to both sides of the input. Default: 0
        dilation: Spacing between kernel elements. Default: 1
        groups: Number of blocked connections from input to output. Default: 1
        bias: If True, adds a learnable bias. Default: True

    Shape:
        - Input: (N, C_in, D, H, W) complex tensor
        - Output: (N, C_out, D_out, H_out, W_out) complex tensor
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int, int]],
        stride: Union[int, tuple[int, int, int]] = 1,
        padding: Union[int, tuple[int, int, int]] = 0,
        dilation: Union[int, tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *self.kernel_size, dtype=torch.cfloat))

        if bias:
            self.bias_r = nn.Parameter(torch.empty(out_channels))
            self.bias_i = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_i', None)

        self.reset_parameters()

    def reset_parameters(self):
        _complex_kaiming_uniform_conv_(self.weight)
        if self.bias_r is not None:
            fan_in = self.weight[0].numel()
            _init_real_bias(self.bias_r, fan_in)
            _init_real_bias(self.bias_i, fan_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(
            x, self.weight, _coupled_bias(self.bias_r, self.bias_i),
            self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, dilation={self.dilation}, '
                f'groups={self.groups}, bias={self.bias_r is not None}')


# ============================================================================
# BICOMPLEX CONVOLUTION LAYERS
# ============================================================================

def _unpack_conv_input(x, input_format):
    """Unpack convolution input to idempotent components (z1, z2).

    For convolution, the bicomplex standard form stores the 4 components
    as separate channel groups: (N, C*4, *spatial) where the channel
    dimension is organized as [a_channels, b_channels, c_channels, d_channels].

    Idempotent form is a tuple (z1, z2) of complex tensors each with shape
    (N, C, *spatial).
    """
    if is_idempotent(x):
        return x
    if isinstance(x, torch.Tensor) and not x.is_complex():
        # Standard form: (N, C*4, *spatial) real tensor
        # Split channel dim into 4 groups of C channels
        n_ch = x.shape[1] // 4
        a, b, c, d = x[:, :n_ch], x[:, n_ch:2*n_ch], x[:, 2*n_ch:3*n_ch], x[:, 3*n_ch:]
        # Idempotent decomposition: z1 = (a+d) + i(b+c), z2 = (a-d) + i(b-c)
        # (matches to_idempotent but operates on channel-first layout)
        z1 = torch.complex(a + d, b + c)
        z2 = torch.complex(a - d, b - c)
        return (z1, z2)
    raise ValueError(
        f"Expected bicomplex conv input: tuple of complex tensors (idempotent) "
        f"or real tensor with shape (N, C*4, *spatial), got {type(x)}")


def _pack_conv_output(z1, z2):
    """Convert idempotent components back to standard form for conv output.

    Inverse of the idempotent decomposition:
        a = (Re(z1) + Re(z2)) / 2
        b = (Im(z1) + Im(z2)) / 2
        c = (Im(z1) - Im(z2)) / 2
        d = (Re(z1) - Re(z2)) / 2

    Returns real tensor of shape (N, C*4, *spatial).
    """
    a = (z1.real + z2.real) / 2
    b = (z1.imag + z2.imag) / 2
    c = (z1.imag - z2.imag) / 2
    d = (z1.real - z2.real) / 2
    return torch.cat([a, b, c, d], dim=1)


class BiComplexConv2d(nn.Module):
    """
    Bicomplex-valued 2D convolution layer.

    Decomposes bicomplex input into two idempotent complex components and
    applies independent complex convolutions to each, mirroring the
    approach used in BiComplexLinear.

    Args:
        in_channels: Number of bicomplex input channels (NOT multiplied by 4)
        out_channels: Number of bicomplex output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 1
        padding: Zero-padding added to both sides. Default: 0
        dilation: Spacing between kernel elements. Default: 1
        groups: Number of blocked connections from input to output. Default: 1
        bias: If True, adds a learnable bias. Default: True
        shared_weights: If True, both branches share the same kernel.
                       Default: False
        input_format: Expected input format ('standard' or 'idempotent').
                      Default: 'standard'
        output_format: Output format ('standard' or 'idempotent').
                       Default: 'standard'

    Shape:
        - Input (standard): (N, C_in*4, H, W) real tensor
        - Input (idempotent): tuple of (N, C_in, H, W) complex tensors
        - Output (standard): (N, C_out*4, H_out, W_out) real tensor
        - Output (idempotent): tuple of (N, C_out, H_out, W_out) complex tensors
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int]] = 0,
        dilation: Union[int, tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        shared_weights: bool = False,
        input_format: Literal['standard', 'idempotent'] = 'standard',
        output_format: Literal['standard', 'idempotent'] = 'standard',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.shared_weights = shared_weights
        self.input_format = input_format
        self.output_format = output_format

        # Branch 1: complex conv kernel + real bias pair
        self.weight1 = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *self.kernel_size, dtype=torch.cfloat))
        if bias:
            self.bias1_r = nn.Parameter(torch.empty(out_channels))
            self.bias1_i = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias1_r', None)
            self.register_parameter('bias1_i', None)

        # Branch 2 (only when not sharing)
        if not shared_weights:
            self.weight2 = nn.Parameter(torch.empty(
                out_channels, in_channels // groups, *self.kernel_size, dtype=torch.cfloat))
            if bias:
                self.bias2_r = nn.Parameter(torch.empty(out_channels))
                self.bias2_i = nn.Parameter(torch.empty(out_channels))
            else:
                self.register_parameter('bias2_r', None)
                self.register_parameter('bias2_i', None)

        self.reset_parameters()

    def reset_parameters(self):
        _complex_kaiming_uniform_conv_(self.weight1)
        if self.bias1_r is not None:
            fan_in = self.weight1[0].numel()
            _init_real_bias(self.bias1_r, fan_in)
            _init_real_bias(self.bias1_i, fan_in)
        if not self.shared_weights:
            _complex_kaiming_uniform_conv_(self.weight2)
            if self.bias2_r is not None:
                fan_in = self.weight2[0].numel()
                _init_real_bias(self.bias2_r, fan_in)
                _init_real_bias(self.bias2_i, fan_in)

    def forward(
        self,
        x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        z1, z2 = _unpack_conv_input(x, self.input_format)

        b1 = _coupled_bias(self.bias1_r, self.bias1_i)
        out1 = F.conv2d(z1, self.weight1, b1,
                        self.stride, self.padding, self.dilation, self.groups)
        if self.shared_weights:
            out2 = F.conv2d(z2, self.weight1, b1,
                            self.stride, self.padding, self.dilation, self.groups)
        else:
            b2 = _coupled_bias(self.bias2_r, self.bias2_i)
            out2 = F.conv2d(z2, self.weight2, b2,
                            self.stride, self.padding, self.dilation, self.groups)

        if self.output_format == 'standard':
            return _pack_conv_output(out1, out2)
        return (out1, out2)

    def extra_repr(self) -> str:
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, dilation={self.dilation}, '
                f'groups={self.groups}, bias={self.bias1_r is not None}, '
                f'shared_weights={self.shared_weights}, '
                f'input_format={self.input_format}, '
                f'output_format={self.output_format}')


class BiComplexConv3d(nn.Module):
    """
    Bicomplex-valued 3D convolution layer.

    Decomposes bicomplex input into two idempotent complex components and
    applies independent complex convolutions to each, mirroring the
    approach used in BiComplexLinear.

    Args:
        in_channels: Number of bicomplex input channels (NOT multiplied by 4)
        out_channels: Number of bicomplex output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 1
        padding: Zero-padding added to both sides. Default: 0
        dilation: Spacing between kernel elements. Default: 1
        groups: Number of blocked connections from input to output. Default: 1
        bias: If True, adds a learnable bias. Default: True
        shared_weights: If True, both branches share the same kernel.
                       Default: False
        input_format: Expected input format ('standard' or 'idempotent').
                      Default: 'standard'
        output_format: Output format ('standard' or 'idempotent').
                       Default: 'standard'

    Shape:
        - Input (standard): (N, C_in*4, D, H, W) real tensor
        - Input (idempotent): tuple of (N, C_in, D, H, W) complex tensors
        - Output (standard): (N, C_out*4, D_out, H_out, W_out) real tensor
        - Output (idempotent): tuple of (N, C_out, D_out, H_out, W_out) complex tensors
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int, int]],
        stride: Union[int, tuple[int, int, int]] = 1,
        padding: Union[int, tuple[int, int, int]] = 0,
        dilation: Union[int, tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        shared_weights: bool = False,
        input_format: Literal['standard', 'idempotent'] = 'standard',
        output_format: Literal['standard', 'idempotent'] = 'standard',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.shared_weights = shared_weights
        self.input_format = input_format
        self.output_format = output_format

        # Branch 1
        self.weight1 = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *self.kernel_size, dtype=torch.cfloat))
        if bias:
            self.bias1_r = nn.Parameter(torch.empty(out_channels))
            self.bias1_i = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias1_r', None)
            self.register_parameter('bias1_i', None)

        # Branch 2
        if not shared_weights:
            self.weight2 = nn.Parameter(torch.empty(
                out_channels, in_channels // groups, *self.kernel_size, dtype=torch.cfloat))
            if bias:
                self.bias2_r = nn.Parameter(torch.empty(out_channels))
                self.bias2_i = nn.Parameter(torch.empty(out_channels))
            else:
                self.register_parameter('bias2_r', None)
                self.register_parameter('bias2_i', None)

        self.reset_parameters()

    def reset_parameters(self):
        _complex_kaiming_uniform_conv_(self.weight1)
        if self.bias1_r is not None:
            fan_in = self.weight1[0].numel()
            _init_real_bias(self.bias1_r, fan_in)
            _init_real_bias(self.bias1_i, fan_in)
        if not self.shared_weights:
            _complex_kaiming_uniform_conv_(self.weight2)
            if self.bias2_r is not None:
                fan_in = self.weight2[0].numel()
                _init_real_bias(self.bias2_r, fan_in)
                _init_real_bias(self.bias2_i, fan_in)

    def forward(
        self,
        x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        z1, z2 = _unpack_conv_input(x, self.input_format)

        b1 = _coupled_bias(self.bias1_r, self.bias1_i)
        out1 = F.conv3d(z1, self.weight1, b1,
                        self.stride, self.padding, self.dilation, self.groups)
        if self.shared_weights:
            out2 = F.conv3d(z2, self.weight1, b1,
                            self.stride, self.padding, self.dilation, self.groups)
        else:
            b2 = _coupled_bias(self.bias2_r, self.bias2_i)
            out2 = F.conv3d(z2, self.weight2, b2,
                            self.stride, self.padding, self.dilation, self.groups)

        if self.output_format == 'standard':
            return _pack_conv_output(out1, out2)
        return (out1, out2)

    def extra_repr(self) -> str:
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, dilation={self.dilation}, '
                f'groups={self.groups}, bias={self.bias1_r is not None}, '
                f'shared_weights={self.shared_weights}, '
                f'input_format={self.input_format}, '
                f'output_format={self.output_format}')
