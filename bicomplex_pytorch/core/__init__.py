"""
Core bicomplex operations and representations.

This module provides the fundamental building blocks for bicomplex number
manipulation including:
- Representation conversions (standard <-> idempotent)
- Arithmetic operations (add, multiply, divide, etc.)
- Tensor operations (matmul, transpose, reshape, etc.)

Example:
    >>> import bicomplex_pytorch.core as bc
    >>> # Create a bicomplex tensor
    >>> z = bc.make_bicomplex(a, b, c, d)
    >>> # Convert to idempotent form for efficient computation
    >>> e1, e2 = bc.to_idempotent(z)
    >>> # Perform multiplication (component-wise in idempotent form)
    >>> result = bc.multiply_idempotent((e1, e2), (w1, w2))
"""

# =============================================================================
# Representations - Format conversions and type checking
# =============================================================================
from .representations import (
    # Type checking
    is_bicomplex,
    is_idempotent,
    # Conversion TO bicomplex (standard form)
    real_to_bicomplex,
    complex_to_bicomplex,
    complex_j_to_bicomplex,
    hyperbolic_to_bicomplex,
    create_bicomplex,
    make_bicomplex,
    # Conversion FROM bicomplex
    bicomplex_to_complex,
    bicomplex_to_real,
    get_components,
    # Idempotent <-> Standard conversions
    to_idempotent,
    from_idempotent,
    to_cartesian,
)

# =============================================================================
# Arithmetic - Mathematical operations on bicomplex numbers
# =============================================================================
from .arithmetic import (
    # Zero divisor handling
    ZeroDivisorError,
    is_zero_divisor,
    has_zero_divisor,
    regularize_zero_divisors,
    # Component extraction (standard form)
    get_real_part,
    get_i_part,
    get_j_part,
    get_ij_part,
    # Projections
    bicomplex_to_complex_i,
    bicomplex_to_complex_j,
    bicomplex_to_hyperbolic,
    # Basic arithmetic (idempotent form)
    add_idempotent,
    subtract_idempotent,
    multiply_idempotent,
    divide_idempotent,
    inverse_idempotent,
    reciprocal_idempotent,
    scalar_multiply_idempotent,
    negate_idempotent,
    # Basic arithmetic (standard form)
    add_standard,
    # Conjugates
    conjugate_i_standard,
    conjugate_j_standard,
    conjugate_total_standard,
    conjugate_idempotent,
    # Norms and moduli
    norm_idempotent,
    modulus,
    modulus_squared,
    d_modulus,
    squared_idempotent_norm,
    # Exponential and logarithm
    exp_idempotent,
    log_idempotent,
    exp_standard,
    log_standard,
    # Powers and roots
    power_idempotent,
    power_standard,
    square_idempotent,
    sqrt_idempotent,
    sqrt_standard,
    bicomplex_power_idempotent,
)

# =============================================================================
# Tensor Operations - PyTorch-like operations for bicomplex tensors
# =============================================================================
from .tensor_ops import (
    # Matrix operations
    matmul,
    bmm,
    transpose,
    einsum,
    # Reductions
    sum,
    mean,
    norm,
    # Shape manipulation
    reshape,
    flatten,
    permute,
    squeeze,
    unsqueeze,
    view,
    # Combining tensors
    cat,
    stack,
    split,
    chunk,
    # Memory operations
    clone,
    detach,
    contiguous,
    expand,
    repeat,
    # Indexing and selection
    gather,
    scatter,
    masked_select,
    masked_fill,
    where,
    index_select,
    take,
)

__all__ = [
    # === Representations ===
    # Type checking
    "is_bicomplex",
    "is_idempotent",
    # Conversion TO bicomplex
    "real_to_bicomplex",
    "complex_to_bicomplex",
    "complex_j_to_bicomplex",
    "hyperbolic_to_bicomplex",
    "create_bicomplex",
    "make_bicomplex",
    # Conversion FROM bicomplex
    "bicomplex_to_complex",
    "bicomplex_to_real",
    "get_components",
    # Idempotent conversions
    "to_idempotent",
    "from_idempotent",
    "to_cartesian",
    # === Arithmetic ===
    # Zero divisors
    "ZeroDivisorError",
    "is_zero_divisor",
    "has_zero_divisor",
    "regularize_zero_divisors",
    # Component extraction
    "get_real_part",
    "get_i_part",
    "get_j_part",
    "get_ij_part",
    # Projections
    "bicomplex_to_complex_i",
    "bicomplex_to_complex_j",
    "bicomplex_to_hyperbolic",
    # Basic arithmetic (idempotent)
    "add_idempotent",
    "subtract_idempotent",
    "multiply_idempotent",
    "divide_idempotent",
    "inverse_idempotent",
    "reciprocal_idempotent",
    "scalar_multiply_idempotent",
    "negate_idempotent",
    # Basic arithmetic (standard)
    "add_standard",
    # Conjugates
    "conjugate_i_standard",
    "conjugate_j_standard",
    "conjugate_total_standard",
    "conjugate_idempotent",
    # Norms
    "norm_idempotent",
    "modulus",
    "modulus_squared",
    "d_modulus",
    "squared_idempotent_norm",
    # Exp/Log
    "exp_idempotent",
    "log_idempotent",
    "exp_standard",
    "log_standard",
    # Powers
    "power_idempotent",
    "power_standard",
    "square_idempotent",
    "sqrt_idempotent",
    "sqrt_standard",
    "bicomplex_power_idempotent",
    # === Tensor Operations ===
    "matmul",
    "bmm",
    "transpose",
    "einsum",
    "sum",
    "mean",
    "norm",
    "reshape",
    "flatten",
    "permute",
    "squeeze",
    "unsqueeze",
    "view",
    "cat",
    "stack",
    "split",
    "chunk",
    "clone",
    "detach",
    "contiguous",
    "expand",
    "repeat",
    "gather",
    "scatter",
    "masked_select",
    "masked_fill",
    "where",
    "index_select",
    "take",
]
