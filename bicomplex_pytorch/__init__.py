# """
# BiComplex PyTorch - Bicomplex number operations for PyTorch
# """
#
# from bicomplex_pytorch.__version__ import __version__
#
# # Import core functionality
# from bicomplex_pytorch.core.representations import (
#     to_idempotent,
#     from_idempotent,
#     # Add other key representation functions
# )
# from bicomplex_pytorch.core.arithmetic import (
#     # Add key arithmetic operations
# )
#
# # Import main nn modules
# from bicomplex_pytorch.nn.modules.linear import BiComplexLinear
# from bicomplex_pytorch.nn.modules.conv import BiComplexConv1d, BiComplexConv2d, BiComplexConv3d
# from bicomplex_pytorch.nn.modules.pooling import BiComplexMaxPool, BiComplexAvgPool
# from bicomplex_pytorch.nn.modules.normalization import BiComplexBatchNorm
# from bicomplex_pytorch.nn.modules.dropout import BiComplexDropout
# from bicomplex_pytorch.nn.activation import (
#     # Add key activation functions
# )
# from bicomplex_pytorch.nn.loss import (
#     # Add key loss functions
# )
#
# # Import functional API
# from bicomplex_pytorch import functional as F
#
# # Import utilities
# from bicomplex_pytorch import utils
#
# __all__ = [
#     "__version__",
#     # Core
#     "to_idempotent",
#     "from_idempotent",
#     # NN Modules
#     "BiComplexLinear",
#     "BiComplexConv1d",
#     "BiComplexConv2d",
#     "BiComplexConv3d",
#     "BiComplexMaxPool",
#     "BiComplexAvgPool",
#     "BiComplexBatchNorm",
#     "BiComplexDropout",
#     # Submodules
#     "F",
#     "utils",
# ]