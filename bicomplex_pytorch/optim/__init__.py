"""
Optimizers for bicomplex neural networks.

This module will provide custom optimizers adapted for bicomplex-valued
neural networks. Currently, standard PyTorch optimizers work well with
bicomplex networks since they operate on the underlying real/complex
parameters.

Note:
    For most use cases, standard PyTorch optimizers (torch.optim.Adam,
    torch.optim.SGD, etc.) work correctly with bicomplex networks.
    Custom bicomplex optimizers may be added for specialized training
    strategies.

Example:
    >>> import torch.optim as optim
    >>> # Standard optimizers work with bicomplex networks
    >>> model = BiComplexLinear(128, 64)
    >>> optimizer = optim.Adam(model.parameters(), lr=0.001)

Planned optimizers:
    - BiComplexAdam: Adam with bicomplex-aware moment estimation
    - BiComplexSGD: SGD with bicomplex gradient handling
    - BiComplexAdamW: AdamW with weight decay for bicomplex
"""

# =============================================================================
# Custom Optimizers (Not yet implemented)
# =============================================================================
# from bicomplex_pytorch.optim.optimizer import (
#     BiComplexAdam,
#     BiComplexSGD,
#     BiComplexAdamW,
#     BiComplexRMSprop,
# )

__all__ = [
    # Planned optimizers
    # "BiComplexAdam",
    # "BiComplexSGD",
    # "BiComplexAdamW",
    # "BiComplexRMSprop",
]
