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
from typing import Optional, Literal, Union
from ...core.representations import to_idempotent, from_idempotent, is_idempotent, is_bicomplex

try:
    from complexPyTorch.complexLayers import ComplexLinear
except ImportError:
    raise ImportError(
        "complexPyTorch is required. Install it with: pip install complexPyTorch"
    )


class BiComplexLinear(nn.Module):
    """
    Bicomplex linear layer using idempotent representation.

    Applies a linear transformation to bicomplex-valued input by
    decomposing into two independent complex-valued branches in
    the idempotent representation.

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

    Attributes:
        shared_weights: Whether weights are shared between branches
        branch1: Complex linear layer for first idempotent component
        branch2: Complex linear layer for second idempotent component
                (only if shared_weights=False)
        complex_layer: Shared complex layer (only if shared_weights=True)

    Note:
        The idempotent representation allows us to process bicomplex
        numbers as two independent complex numbers, avoiding issues
        with zero divisors in standard bicomplex arithmetic.
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

        # Note: complexPyTorch's ComplexLinear doesn't support bias parameter
        # It always includes bias by default
        if shared_weights:
            self.complex_layer = ComplexLinear(in_features, out_features)
        else:
            self.branch1 = ComplexLinear(in_features, out_features)
            self.branch2 = ComplexLinear(in_features, out_features)

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
        # Handle input conversion
        if self.input_format == 'standard':
            if is_idempotent(x):
                # User passed idempotent but we expect standard - auto-convert
                z1, z2 = x
            elif is_bicomplex(x):
                z1, z2 = to_idempotent(x)
            else:
                raise ValueError(
                    f"Expected standard form input with shape (..., {self.in_features}, 4), "
                    f"got shape {x.shape}"
                )
        else:  # idempotent
            if is_idempotent(x):
                z1, z2 = x
            elif is_bicomplex(x):
                # User passed standard but we expect idempotent - auto-convert
                z1, z2 = to_idempotent(x)
            else:
                raise ValueError("Expected idempotent form input (tuple of complex tensors)")

        # Process through complex branches
        if self.shared_weights:
            out1 = self.complex_layer(z1)
            out2 = self.complex_layer(z2)
        else:
            out1 = self.branch1(z1)
            out2 = self.branch2(z2)

        # Handle output conversion
        if self.output_format == 'standard':
            return from_idempotent(out1, out2)
        else:
            return (out1, out2)

    def extra_repr(self) -> str:
        """String representation for print()."""
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'shared_weights={self.shared_weights}, '
                f'input_format={self.input_format}, '
                f'output_format={self.output_format}')


class BiComplexLinearFull(nn.Module):
    """
    Full bicomplex linear layer with cross-component interactions.

    This implements the most general bicomplex-linear transformation:
        (z₁', z₂') = (W₁₁·z₁ + W₁₂·z₂, W₂₁·z₁ + W₂₂·z₂)

    In matrix form:
        [z₁']   [W₁₁  W₁₂] [z₁]
        [z₂'] = [W₂₁  W₂₂] [z₂]

    This allows learning transformations that mix the idempotent components,
    which is not possible with the standard BiComplexLinear layer.

    Mathematical Note:
    ==================
    In standard bicomplex linear algebra, a BC-linear map L: BC → BC has the form:
        L(z) = α·z + β·z̄

    where α, β ∈ BC and z̄ is the bicomplex conjugate. The cross-component
    terms W₁₂ and W₂₁ capture the β·z̄ part of this transformation.

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

        # Four weight matrices for full bicomplex linear transformation
        # W₁₁: e1 -> e1 (self-interaction)
        # W₁₂: e2 -> e1 (cross-interaction)
        # W₂₁: e1 -> e2 (cross-interaction)
        # W₂₂: e2 -> e2 (self-interaction)
        # Note: complexPyTorch's ComplexLinear always includes bias, so we handle
        # bias separately below to have proper control
        self.W11 = ComplexLinear(in_features, out_features)
        self.W12 = ComplexLinear(in_features, out_features)
        self.W21 = ComplexLinear(in_features, out_features)
        self.W22 = ComplexLinear(in_features, out_features)
        # Zero out the default biases from ComplexLinear since we manage bias separately
        with torch.no_grad():
            self.W11.fc_r.bias.zero_()
            self.W11.fc_i.bias.zero_()
            self.W12.fc_r.bias.zero_()
            self.W12.fc_i.bias.zero_()
            self.W21.fc_r.bias.zero_()
            self.W21.fc_i.bias.zero_()
            self.W22.fc_r.bias.zero_()
            self.W22.fc_i.bias.zero_()

        # Biases (one for each output component)
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
        # Handle input conversion
        if self.input_format == 'standard':
            if is_idempotent(x):
                z1, z2 = x
            elif is_bicomplex(x):
                z1, z2 = to_idempotent(x)
            else:
                raise ValueError(
                    f"Expected standard form input with shape (..., {self.in_features}, 4)"
                )
        else:
            if is_idempotent(x):
                z1, z2 = x
            elif is_bicomplex(x):
                z1, z2 = to_idempotent(x)
            else:
                raise ValueError("Expected idempotent form input")

        # Full transformation with cross-terms
        out1 = self.W11(z1) + self.W12(z2)
        out2 = self.W21(z1) + self.W22(z2)

        # Add biases
        if self.bias1 is not None:
            out1 = out1 + self.bias1
            out2 = out2 + self.bias2

        # Handle output conversion
        if self.output_format == 'standard':
            return from_idempotent(out1, out2)
        else:
            return (out1, out2)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias1 is not None}, '
                f'input_format={self.input_format}, '
                f'output_format={self.output_format}')


class BiComplexLinearDiagonal(nn.Module):
    """
    Diagonal bicomplex linear layer (no cross-component interaction).

    This is equivalent to BiComplexLinear with shared_weights=False,
    but implemented using native PyTorch complex tensors for efficiency.

    The transformation is:
        (z₁', z₂') = (W₁·z₁ + b₁, W₂·z₂ + b₂)

    This is the most common configuration for bicomplex neural networks,
    balancing expressivity with computational efficiency.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds learnable bias. Default: True
        input_format: Expected input format
        output_format: Output format
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

        # Weight matrices (complex-valued)
        self.weight1 = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.cfloat) / (in_features ** 0.5)
        )
        self.weight2 = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.cfloat) / (in_features ** 0.5)
        )

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
        """Forward pass with diagonal (independent) transformations."""
        # Handle input conversion
        if self.input_format == 'standard':
            if is_idempotent(x):
                z1, z2 = x
            elif is_bicomplex(x):
                z1, z2 = to_idempotent(x)
            else:
                raise ValueError("Expected standard form input")
        else:
            if is_idempotent(x):
                z1, z2 = x
            elif is_bicomplex(x):
                z1, z2 = to_idempotent(x)
            else:
                raise ValueError("Expected idempotent form input")

        # Linear transformations using torch.nn.functional.linear
        out1 = torch.nn.functional.linear(z1, self.weight1, self.bias1)
        out2 = torch.nn.functional.linear(z2, self.weight2, self.bias2)

        # Handle output conversion
        if self.output_format == 'standard':
            return from_idempotent(out1, out2)
        else:
            return (out1, out2)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias1 is not None}')
