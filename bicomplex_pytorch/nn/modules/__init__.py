"""
Bicomplex neural network modules.

This subpackage contains stateful nn.Module classes for building
bicomplex neural networks, following PyTorch's nn module conventions.

Implemented:
- Linear layers (BiComplexLinear, BiComplexLinearFull, BiComplexLinearDiagonal)

Planned (stubs):
- Convolution layers
- Pooling layers
- Normalization layers
- Dropout layers
- Attention layers
"""

# =============================================================================
# Linear Layers (Implemented)
# =============================================================================
from bicomplex_pytorch.nn.modules.linear import (
    BiComplexLinear,
    BiComplexLinearFull,
    BiComplexLinearDiagonal,
)

# =============================================================================
# Convolution Layers (Not yet implemented)
# =============================================================================
# from bicomplex_pytorch.nn.modules.conv import (
#     BiComplexConv1d,
#     BiComplexConv2d,
#     BiComplexConv3d,
#     BiComplexConvTranspose1d,
#     BiComplexConvTranspose2d,
#     BiComplexConvTranspose3d,
# )

# =============================================================================
# Pooling Layers (Not yet implemented)
# =============================================================================
# from bicomplex_pytorch.nn.modules.pooling import (
#     BiComplexMaxPool1d,
#     BiComplexMaxPool2d,
#     BiComplexMaxPool3d,
#     BiComplexAvgPool1d,
#     BiComplexAvgPool2d,
#     BiComplexAvgPool3d,
#     BiComplexAdaptiveMaxPool2d,
#     BiComplexAdaptiveAvgPool2d,
# )

# =============================================================================
# Normalization Layers (Not yet implemented)
# =============================================================================
# from bicomplex_pytorch.nn.modules.normalization import (
#     BiComplexBatchNorm1d,
#     BiComplexBatchNorm2d,
#     BiComplexBatchNorm3d,
#     BiComplexLayerNorm,
#     BiComplexGroupNorm,
#     BiComplexInstanceNorm1d,
#     BiComplexInstanceNorm2d,
# )

# =============================================================================
# Dropout Layers (Not yet implemented)
# =============================================================================
# from bicomplex_pytorch.nn.modules.dropout import (
#     BiComplexDropout,
#     BiComplexDropout2d,
#     BiComplexDropout3d,
#     BiComplexAlphaDropout,
# )

# =============================================================================
# Attention Layers (Not yet implemented)
# =============================================================================
# from bicomplex_pytorch.nn.modules.attention import (
#     BiComplexMultiHeadAttention,
#     BiComplexSelfAttention,
# )

__all__ = [
    # Linear (implemented)
    "BiComplexLinear",
    "BiComplexLinearFull",
    "BiComplexLinearDiagonal",
    # Convolution (planned)
    # "BiComplexConv1d",
    # "BiComplexConv2d",
    # "BiComplexConv3d",
    # "BiComplexConvTranspose1d",
    # "BiComplexConvTranspose2d",
    # "BiComplexConvTranspose3d",
    # Pooling (planned)
    # "BiComplexMaxPool1d",
    # "BiComplexMaxPool2d",
    # "BiComplexMaxPool3d",
    # "BiComplexAvgPool1d",
    # "BiComplexAvgPool2d",
    # "BiComplexAvgPool3d",
    # "BiComplexAdaptiveMaxPool2d",
    # "BiComplexAdaptiveAvgPool2d",
    # Normalization (planned)
    # "BiComplexBatchNorm1d",
    # "BiComplexBatchNorm2d",
    # "BiComplexBatchNorm3d",
    # "BiComplexLayerNorm",
    # "BiComplexGroupNorm",
    # "BiComplexInstanceNorm1d",
    # "BiComplexInstanceNorm2d",
    # Dropout (planned)
    # "BiComplexDropout",
    # "BiComplexDropout2d",
    # "BiComplexDropout3d",
    # "BiComplexAlphaDropout",
    # Attention (planned)
    # "BiComplexMultiHeadAttention",
    # "BiComplexSelfAttention",
]
