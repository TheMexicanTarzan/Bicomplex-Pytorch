"""
Neural network modules for bicomplex numbers.

This module provides PyTorch-style nn.Module classes for building
bicomplex neural networks. It mirrors the structure of torch.nn:

- nn.modules: Layer classes (Linear, Conv, etc.)
- nn.activation: Activation module classes (planned)
- nn.loss: Loss module classes (planned)
- nn.init: Weight initialization functions (planned)

Example:
    >>> import bicomplex_pytorch.nn as bcnn
    >>> # Create a bicomplex linear layer
    >>> layer = bcnn.BiComplexLinear(128, 64)
    >>> output = layer(input)
"""

# =============================================================================
# Modules - Layer classes
# =============================================================================
from .modules import (
    # Linear layers
    BiComplexLinear,
    BiComplexLinearFull,
)

# =============================================================================
# Activation Modules (Not yet implemented)
# =============================================================================
# Activation functions are available in bicomplex_pytorch.functional
# Module wrappers will be added in nn.activation when implemented

# =============================================================================
# Loss Modules (Not yet implemented)
# =============================================================================
# Loss functions are available in bicomplex_pytorch.functional
# Module wrappers will be added in nn.loss when implemented

# =============================================================================
# Initialization Functions (Not yet implemented)
# =============================================================================
# Initialization utilities are available in bicomplex_pytorch.functional.utils
# Dedicated init module will be added in nn.init when implemented

__all__ = [
    # === Modules ===
    # Linear layers (implemented)
    "BiComplexLinear",
    "BiComplexLinearFull",
    # Convolution layers (planned)
    # "BiComplexConv1d",
    # "BiComplexConv2d",
    # "BiComplexConv3d",
    # Pooling layers (planned)
    # "BiComplexMaxPool1d",
    # "BiComplexMaxPool2d",
    # "BiComplexAvgPool2d",
    # Normalization layers (planned)
    # "BiComplexBatchNorm1d",
    # "BiComplexBatchNorm2d",
    # "BiComplexLayerNorm",
    # Dropout layers (planned)
    # "BiComplexDropout",
    # "BiComplexDropout2d",
    # Attention layers (planned)
    # "BiComplexMultiHeadAttention",
]
