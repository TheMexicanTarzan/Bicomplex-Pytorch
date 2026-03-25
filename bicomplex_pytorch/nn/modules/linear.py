"""
Bicomplex linear layers.

This module implements linear transformations for bicomplex-valued
neural networks using the idempotent representation.

================================================================================
INPUT/OUTPUT CONVENTIONS
================================================================================

This module supports TWO input formats:

1. STANDARD FORM: Tensor of shape (..., in_features, 4)
   - Last dimension contains [a, b, c, d] representing z = a + bi + cj + dij
   - This is the default input format for end-users

2. IDEMPOTENT FORM: Tuple (e1, e2) of complex tensors
   - Each tensor has shape (..., in_features)
   - Used internally for efficient computation

By default, layers accept standard form input and return standard form output.
Set `input_format='idempotent'` or `output_format='idempotent'` to change this.

================================================================================
WEIGHT CONFIGURATIONS
================================================================================

1. SHARED WEIGHTS (shared_weights=True):
   - Same transformation applied to both idempotent components
   - Equivalent to: z' = W·z (diagonal block matrix)
   - Parameters: N complex weights
   - Use when: complex-like behavior is desired

2. INDEPENDENT WEIGHTS (shared_weights=False, default):
   - Different transformations for each component
   - Equivalent to: (z₁', z₂') = (W₁·z₁, W₂·z₂)
   - Parameters: 2N complex weights
   - Use when: maximum expressivity is needed

3. FULL BICOMPLEX (BiComplexLinearFull):
   - Includes cross-component interactions
   - Equivalent to: (z₁', z₂') = (W₁₁·z₁ + W₁₂·z₂, W₂₁·z₁ + W₂₂·z₂)
   - Parameters: 4N complex weights
   - Use when: full bicomplex linear map is required

================================================================================
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Union
from ...core.representations import to_idempotent, from_idempotent, is_idempotent, is_bicomplex


def _unpack_input(x, input_format, in_features):
    """Unpack input to idempotent components (z1, z2)."""
    if is_idempotent(x):
        return x
    if is_bicomplex(x):
        return to_idempotent(x)
    if input_format == 'standard':
        raise ValueError(
            f"Expected standard form input with shape (..., {in_features}, 4), "
            f"got shape {x.shape}"
        )
    raise ValueError("Expected idempotent form input (tuple of complex tensors)")


def _apply_complex(z, W_r, W_i, b_r, b_i):
    """Apply a complex linear transformation using two real linear operations.

    Reproduces the computation of complexPyTorch's apply_complex:
        output = (W_r @ z.real - W_i @ z.imag + b_r - b_i)
               + j*(W_r @ z.imag + W_i @ z.real + b_r + b_i)

    This structure ensures each real bias (b_r, b_i) participates in both
    the real and imaginary output paths, giving them coupled gradients with
    a richer error signal than decoupled complex bias.
    """
    real_part = F.linear(z.real, W_r, b_r) - F.linear(z.imag, W_i, b_i)
    imag_part = F.linear(z.imag, W_r, b_r) + F.linear(z.real, W_i, b_i)
    return real_part.to(torch.cfloat) + 1j * imag_part.to(torch.cfloat)


def _init_real_weight_and_bias(weight, bias):
    """Initialize real weight and bias matching PyTorch nn.Linear defaults.

    Uses Kaiming uniform with a=sqrt(5) for weights and
    uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) for bias.
    """
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in = weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)


class BiComplexLinear(nn.Module):
    """
    Bicomplex linear layer using idempotent representation.

    Applies a linear transformation to bicomplex-valued input by
    decomposing into two independent complex-valued branches in
    the idempotent representation, using the apply_complex pattern
    (two real linear layers per complex branch).

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
              Default: True
        shared_weights: If True, both branches share the same weights.
                       If False, each branch has independent weights.
                       Default: False
        input_format: Expected input format ('standard' or 'idempotent').
                      Default: 'standard'
        output_format: Output format ('standard' or 'idempotent').
                       Default: 'standard'

    Shape:
        - Input (standard): (N, *, in_features, 4) where * means any number of
                 additional dimensions and 4 represents bicomplex components
        - Input (idempotent): tuple of (N, *, in_features) complex tensors
        - Output (standard): (N, *, out_features, 4)
        - Output (idempotent): tuple of (N, *, out_features) complex tensors
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            shared_weights: bool = False,
            input_format: Literal['standard', 'idempotent'] = 'standard',
            output_format: Literal['standard', 'idempotent'] = 'standard'
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared_weights = shared_weights
        self.input_format = input_format
        self.output_format = output_format

        # Branch 1: real and imaginary sub-layers (always present)
        self.weight1_r = nn.Parameter(torch.empty(out_features, in_features))
        self.weight1_i = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias1_r = nn.Parameter(torch.empty(out_features))
            self.bias1_i = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias1_r', None)
            self.register_parameter('bias1_i', None)

        # Branch 2 (only when not sharing)
        if not shared_weights:
            self.weight2_r = nn.Parameter(torch.empty(out_features, in_features))
            self.weight2_i = nn.Parameter(torch.empty(out_features, in_features))
            if bias:
                self.bias2_r = nn.Parameter(torch.empty(out_features))
                self.bias2_i = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter('bias2_r', None)
                self.register_parameter('bias2_i', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters matching Kaiming uniform (PyTorch nn.Linear default)."""
        _init_real_weight_and_bias(self.weight1_r, self.bias1_r)
        _init_real_weight_and_bias(self.weight1_i, self.bias1_i)
        if not self.shared_weights:
            _init_real_weight_and_bias(self.weight2_r, self.bias2_r)
            _init_real_weight_and_bias(self.weight2_i, self.bias2_i)

    def forward(
        self,
        x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the bicomplex linear layer.

        Args:
            x: Bicomplex input tensor. Format depends on input_format:
               - 'standard': tensor of shape (..., in_features, 4)
               - 'idempotent': tuple (e1, e2) of complex tensors

        Returns:
            Bicomplex output tensor. Format depends on output_format:
            - 'standard': tensor of shape (..., out_features, 4)
            - 'idempotent': tuple (e1, e2) of complex tensors
        """
        z1, z2 = _unpack_input(x, self.input_format, self.in_features)

        out1 = _apply_complex(z1, self.weight1_r, self.weight1_i,
                              self.bias1_r, self.bias1_i)
        if self.shared_weights:
            out2 = _apply_complex(z2, self.weight1_r, self.weight1_i,
                                  self.bias1_r, self.bias1_i)
        else:
            out2 = _apply_complex(z2, self.weight2_r, self.weight2_i,
                                  self.bias2_r, self.bias2_i)

        if self.output_format == 'standard':
            return from_idempotent(out1, out2)
        return (out1, out2)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias1_r is not None}, '
                f'shared_weights={self.shared_weights}, '
                f'input_format={self.input_format}, '
                f'output_format={self.output_format}')


class BiComplexLinearFull(nn.Module):
    """
    Full bicomplex linear layer with cross-component interactions.

    This implements the most general bicomplex-linear transformation:
        (z₁', z₂') = (W₁₁·z₁ + W₁₂·z₂ + b₁, W₂₁·z₁ + W₂₂·z₂ + b₂)

    In matrix form:
        [z₁']   [W₁₁  W₁₂] [z₁]
        [z₂'] = [W₂₁  W₂₂] [z₂]

    This allows learning transformations that mix the idempotent components,
    which is not possible with the standard BiComplexLinear layer.

    Mathematical Note:
    ==================
    In standard bicomplex linear algebra, a BC-linear map L: BC -> BC has the form:
        L(z) = alpha*z + beta*z_bar

    where alpha, beta are in BC and z_bar is the bicomplex conjugate. The cross-component
    terms W₁₂ and W₂₁ capture the beta*z_bar part of this transformation.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds learnable bias. Default: True
        input_format: Expected input format ('standard' or 'idempotent')
        output_format: Output format ('standard' or 'idempotent')

    Shape:
        - Input (standard): (..., in_features, 4)
        - Input (idempotent): tuple of (..., in_features) complex tensors
        - Output (standard): (..., out_features, 4)
        - Output (idempotent): tuple of (..., out_features) complex tensors

    Note:
        This layer has 4x the parameters of BiComplexLinear with shared_weights=True.
        Use when the task requires full bicomplex linear expressivity.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            input_format: Literal['standard', 'idempotent'] = 'standard',
            output_format: Literal['standard', 'idempotent'] = 'standard'
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_format = input_format
        self.output_format = output_format

        # Four complex sub-layers, each with real/imag weight + bias pairs.
        # Internal biases are zeroed at init but learnable, matching the old
        # ComplexLinear behaviour where each sub-layer carried a learnable bias.
        for name in ('W11', 'W12', 'W21', 'W22'):
            self.register_parameter(
                f'{name}_r', nn.Parameter(torch.empty(out_features, in_features)))
            self.register_parameter(
                f'{name}_i', nn.Parameter(torch.empty(out_features, in_features)))
            self.register_parameter(
                f'{name}_br', nn.Parameter(torch.zeros(out_features)))
            self.register_parameter(
                f'{name}_bi', nn.Parameter(torch.zeros(out_features)))

        # Explicit output biases
        if bias:
            self.bias1 = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat))
            self.bias2 = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weight parameters matching Kaiming uniform (PyTorch nn.Linear default)."""
        for name in ('W11', 'W12', 'W21', 'W22'):
            _init_real_weight_and_bias(getattr(self, f'{name}_r'), bias=None)
            _init_real_weight_and_bias(getattr(self, f'{name}_i'), bias=None)

    def forward(
        self,
        x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with cross-component interactions.

        Computes: (z₁', z₂') = (W₁₁·z₁ + W₁₂·z₂ + b₁, W₂₁·z₁ + W₂₂·z₂ + b₂)
        """
        z1, z2 = _unpack_input(x, self.input_format, self.in_features)

        out1 = (_apply_complex(z1, self.W11_r, self.W11_i, self.W11_br, self.W11_bi) +
                _apply_complex(z2, self.W12_r, self.W12_i, self.W12_br, self.W12_bi))
        out2 = (_apply_complex(z1, self.W21_r, self.W21_i, self.W21_br, self.W21_bi) +
                _apply_complex(z2, self.W22_r, self.W22_i, self.W22_br, self.W22_bi))

        if self.bias1 is not None:
            out1 = out1 + self.bias1
            out2 = out2 + self.bias2

        if self.output_format == 'standard':
            return from_idempotent(out1, out2)
        return (out1, out2)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias1 is not None}, '
                f'input_format={self.input_format}, '
                f'output_format={self.output_format}')
