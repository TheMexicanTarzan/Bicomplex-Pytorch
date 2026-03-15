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


class BiComplexLinear(nn.Module):
    """
    Bicomplex linear layer using idempotent representation.

    Applies a linear transformation to bicomplex-valued input by
    decomposing into two independent complex-valued branches in
    the idempotent representation, using native PyTorch complex tensors.

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

        # Branch 1 weights (always present)
        self.weight1 = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.cfloat) / (in_features ** 0.5)
        )
        if bias:
            self.bias1 = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat))
        else:
            self.register_parameter('bias1', None)

        # Branch 2 weights (only when not sharing)
        if not shared_weights:
            self.weight2 = nn.Parameter(
                torch.randn(out_features, in_features, dtype=torch.cfloat) / (in_features ** 0.5)
            )
            if bias:
                self.bias2 = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat))
            else:
                self.register_parameter('bias2', None)

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

        out1 = F.linear(z1, self.weight1, self.bias1)
        if self.shared_weights:
            out2 = F.linear(z2, self.weight1, self.bias1)
        else:
            out2 = F.linear(z2, self.weight2, self.bias2)

        if self.output_format == 'standard':
            return from_idempotent(out1, out2)
        return (out1, out2)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias1 is not None}, '
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

        scale = in_features ** 0.5

        # Four weight matrices for full bicomplex linear transformation
        self.W11 = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.cfloat) / scale)
        self.W12 = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.cfloat) / scale)
        self.W21 = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.cfloat) / scale)
        self.W22 = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.cfloat) / scale)

        if bias:
            self.bias1 = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat))
            self.bias2 = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)

    def forward(
        self,
        x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with cross-component interactions.

        Computes: (z₁', z₂') = (W₁₁·z₁ + W₁₂·z₂ + b₁, W₂₁·z₁ + W₂₂·z₂ + b₂)
        """
        z1, z2 = _unpack_input(x, self.input_format, self.in_features)

        out1 = F.linear(z1, self.W11) + F.linear(z2, self.W12)
        out2 = F.linear(z1, self.W21) + F.linear(z2, self.W22)

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
