"""
Bicomplex linear layers.

This module implements linear transformations for bicomplex-valued
neural networks using the idempotent representation.
"""
import torch
import torch.nn as nn
from typing import Optional
from ...core.representations import to_idempotent, from_idempotent

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

    Shape:
        - Input: (N, *, in_features, 4) where * means any number of
                 additional dimensions and 4 represents bicomplex components
        - Output: (N, *, out_features, 4)

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
            shared_weights: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared_weights = shared_weights

        if shared_weights:
            self.complex_layer = ComplexLinear(in_features, out_features, bias)
        else:
            self.branch1 = ComplexLinear(in_features, out_features, bias)
            self.branch2 = ComplexLinear(in_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the bicomplex linear layer.

        Args:
            x: Bicomplex input tensor of shape (..., in_features, 4)

        Returns:
            Bicomplex output tensor of shape (..., out_features, 4)
        """
        # Transform to idempotent representation
        z1, z2 = to_idempotent(x)

        # Process through complex branches
        if self.shared_weights:
            out1 = self.complex_layer(z1)
            out2 = self.complex_layer(z2)
        else:
            out1 = self.branch1(z1)
            out2 = self.branch2(z2)

        # Transform back to bicomplex representation
        return from_idempotent(out1, out2)

    def extra_repr(self) -> str:
        """String representation for print()."""
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'shared_weights={self.shared_weights}')