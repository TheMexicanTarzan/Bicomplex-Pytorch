"""
BiComplex PyTorch - Bicomplex number operations for PyTorch.

A comprehensive library for bicomplex-valued neural networks built on PyTorch.
Bicomplex numbers extend complex numbers with a second imaginary unit, enabling
richer representations for signal processing, quantum computing, and more.

Quick Start:
    >>> import bicomplex_pytorch as bc
    >>> import bicomplex_pytorch.functional as F
    >>> import bicomplex_pytorch.nn as nn
    >>>
    >>> # Create a bicomplex tensor
    >>> z = bc.make_bicomplex(a, b, c, d)  # z = a + bi + cj + dij
    >>>
    >>> # Convert to idempotent form for computation
    >>> e1, e2 = bc.to_idempotent(z)
    >>>
    >>> # Use functional API
    >>> output = F.bicomplex_relu((e1, e2))
    >>>
    >>> # Build neural networks
    >>> layer = nn.BiComplexLinear(128, 64)

Modules:
    - core: Fundamental bicomplex operations (arithmetic, representations, tensor ops)
    - functional: Stateless functions (activations, pooling, normalization, etc.)
    - nn: Neural network modules (layers, losses, etc.)
    - optim: Optimizers (planned)
    - utils: Utility functions (planned)

Representations:
    - Standard form: tensor of shape (..., 4) with [a, b, c, d] for z = a + bi + cj + dij
    - Idempotent form: tuple (e1, e2) of complex tensors for efficient computation
"""

from bicomplex_pytorch.__version__ import __version__

# =============================================================================
# Core - Representations (most commonly used at top level)
# =============================================================================
from bicomplex_pytorch.core.representations import (
    # Type checking
    is_bicomplex,
    is_idempotent,
    # Format conversions
    to_idempotent,
    from_idempotent,
    to_cartesian,
    # Creation functions
    create_bicomplex,
    make_bicomplex,
    real_to_bicomplex,
    complex_to_bicomplex,
    # Component extraction
    get_components,
)

# =============================================================================
# Core - Arithmetic (key operations at top level)
# =============================================================================
from bicomplex_pytorch.core.arithmetic import (
    # Zero divisor handling
    ZeroDivisorError,
    is_zero_divisor,
    # Basic operations
    add_idempotent,
    multiply_idempotent,
    divide_idempotent,
    conjugate_idempotent,
    # Norms
    modulus,
    modulus_squared,
    # Transcendental functions
    exp_idempotent,
    log_idempotent,
    sqrt_idempotent,
    power_idempotent,
)

# =============================================================================
# Core - Tensor Operations (key operations at top level)
# =============================================================================
from bicomplex_pytorch.core.tensor_ops import (
    matmul,
    bmm,
    transpose,
)

# =============================================================================
# NN Modules (most commonly used layers at top level)
# =============================================================================
from bicomplex_pytorch.nn import (
    BiComplexLinear,
    BiComplexLinearFull,
    BiComplexLinearDiagonal,
)

# =============================================================================
# Submodules (import as namespace)
# =============================================================================
from bicomplex_pytorch import core
from bicomplex_pytorch import functional
from bicomplex_pytorch import nn
from bicomplex_pytorch import optim
from bicomplex_pytorch import utils

# Convenient alias for functional
F = functional

__all__ = [
    # Version
    "__version__",
    # === Core - Representations ===
    "is_bicomplex",
    "is_idempotent",
    "to_idempotent",
    "from_idempotent",
    "to_cartesian",
    "create_bicomplex",
    "make_bicomplex",
    "real_to_bicomplex",
    "complex_to_bicomplex",
    "get_components",
    # === Core - Arithmetic ===
    "ZeroDivisorError",
    "is_zero_divisor",
    "add_idempotent",
    "multiply_idempotent",
    "divide_idempotent",
    "conjugate_idempotent",
    "modulus",
    "modulus_squared",
    "exp_idempotent",
    "log_idempotent",
    "sqrt_idempotent",
    "power_idempotent",
    # === Core - Tensor Ops ===
    "matmul",
    "bmm",
    "transpose",
    # === NN Modules ===
    "BiComplexLinear",
    "BiComplexLinearFull",
    "BiComplexLinearDiagonal",
    # === Submodules ===
    "core",
    "functional",
    "F",
    "nn",
    "optim",
    "utils",
]
