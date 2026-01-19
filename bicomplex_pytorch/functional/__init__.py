"""
Functional API for bicomplex neural network operations.

This module provides stateless functions for bicomplex tensor operations,
following the PyTorch functional API convention (torch.nn.functional).

All functions operate on bicomplex tensors in idempotent form (e1, e2).

Example:
    >>> import bicomplex_pytorch.functional as F
    >>> # Apply ReLU activation
    >>> output = F.bicomplex_relu(input)
    >>> # Apply linear transformation
    >>> output = F.bicomplex_linear(input, weight, bias)
"""

# =============================================================================
# Activation Functions
# =============================================================================
from bicomplex_pytorch.functional.activation import (
    # Standard activations (component-wise)
    bicomplex_relu,
    bicomplex_leaky_relu,
    bicomplex_elu,
    bicomplex_selu,
    bicomplex_gelu,
    bicomplex_sigmoid,
    bicomplex_tanh,
    bicomplex_softplus,
    bicomplex_softsign,
    bicomplex_hardtanh,
    bicomplex_softmax,
    bicomplex_log_softmax,
    # Modern activations
    bicomplex_swish,
    bicomplex_mish,
    bicomplex_prelu,
    bicomplex_rrelu,
    bicomplex_threshold,
    # Bicomplex-specific activations
    bicomplex_modulus_activation,
    bicomplex_normalize,
    bicomplex_phase_activation,
    # Holomorphic activations
    bicomplex_exp_activation,
    bicomplex_bounded_exp_activation,
    bicomplex_sin_activation,
    bicomplex_cos_activation,
    bicomplex_sinh_activation,
    bicomplex_cosh_activation,
    bicomplex_tanh_holomorphic,
    # Polynomial activations
    bicomplex_polynomial_activation,
    bicomplex_learnable_polynomial_activation,
    bicomplex_rational_activation,
    # Split activations
    bicomplex_split_activation,
)

# =============================================================================
# Linear Operations
# =============================================================================
from bicomplex_pytorch.functional.linear import (
    bicomplex_linear,
    bicomplex_bilinear,
    bicomplex_matmul,
    bicomplex_addmm,
    bicomplex_addmv,
    bicomplex_addr,
)

# =============================================================================
# Convolution Operations
# =============================================================================
from bicomplex_pytorch.functional.conv import (
    bicomplex_conv1d,
    bicomplex_conv2d,
    bicomplex_conv3d,
    bicomplex_conv_transpose1d,
    bicomplex_conv_transpose2d,
    bicomplex_conv_transpose3d,
    bicomplex_unfold,
    bicomplex_fold,
)

# =============================================================================
# Pooling Operations
# =============================================================================
from bicomplex_pytorch.functional.pooling import (
    # Max pooling
    bicomplex_max_pool1d,
    bicomplex_max_pool2d,
    bicomplex_max_pool3d,
    # Average pooling
    bicomplex_avg_pool1d,
    bicomplex_avg_pool2d,
    bicomplex_avg_pool3d,
    # Adaptive max pooling
    bicomplex_adaptive_max_pool1d,
    bicomplex_adaptive_max_pool2d,
    bicomplex_adaptive_max_pool3d,
    # Adaptive average pooling
    bicomplex_adaptive_avg_pool1d,
    bicomplex_adaptive_avg_pool2d,
    bicomplex_adaptive_avg_pool3d,
    # LP pooling
    bicomplex_lp_pool1d,
    bicomplex_lp_pool2d,
    # Unpooling
    bicomplex_max_unpool1d,
    bicomplex_max_unpool2d,
    bicomplex_max_unpool3d,
    # Fractional pooling
    bicomplex_fractional_max_pool2d,
    bicomplex_fractional_max_pool3d,
)

# =============================================================================
# Normalization Operations
# =============================================================================
from bicomplex_pytorch.functional.normalization import (
    bicomplex_batch_norm,
    bicomplex_layer_norm,
    bicomplex_instance_norm,
    bicomplex_group_norm,
    bicomplex_local_response_norm,
    bicomplex_normalize as bicomplex_lp_normalize,  # Renamed to avoid conflict
    bicomplex_rms_norm,
    bicomplex_weight_norm,
    bicomplex_spectral_norm,
)

# =============================================================================
# Dropout Operations
# =============================================================================
from bicomplex_pytorch.functional.dropout import (
    bicomplex_dropout,
    bicomplex_dropout1d,
    bicomplex_dropout2d,
    bicomplex_dropout3d,
    bicomplex_alpha_dropout,
    bicomplex_feature_alpha_dropout,
    bicomplex_component_dropout,
    bicomplex_zoneout,
    bicomplex_dropconnect,
    bicomplex_variational_dropout,
    bicomplex_spatial_dropout,
    bicomplex_gaussian_dropout,
    bicomplex_concrete_dropout,
)

# =============================================================================
# Loss Functions
# =============================================================================
from bicomplex_pytorch.functional.loss import (
    mse_loss,
    l1_loss,
    smooth_l1_loss,
    component_mse_loss,
    cosine_embedding_loss,
    triplet_margin_loss,
    hinge_embedding_loss,
    kl_divergence_loss,
    margin_ranking_loss,
    cauchy_loss,
)

# =============================================================================
# Attention Mechanisms
# =============================================================================
from bicomplex_pytorch.functional.attention import (
    scaled_dot_product_attention,
    multi_head_attention,
    additive_attention,
    self_attention,
    cross_attention,
    multiplicative_attention,
    local_attention,
)

# =============================================================================
# Utility Functions
# =============================================================================
from bicomplex_pytorch.functional.utils import (
    # Tensor creation
    zeros_idempotent,
    ones_idempotent,
    randn_idempotent,
    rand_idempotent,
    randn_standard,
    # Weight initialization
    xavier_uniform_idempotent,
    xavier_normal_idempotent,
    kaiming_uniform_idempotent,
    kaiming_normal_idempotent,
    uniform_idempotent,
    normal_idempotent,
    # Conversion utilities
    real_to_bicomplex_idempotent,
    complex_to_bicomplex_idempotent,
    tensor_to_bicomplex,
    # Component extraction
    extract_components_standard,
    extract_components_idempotent,
    compute_statistics_idempotent,
    # Training utilities
    clip_grad_norm_idempotent,
    zero_grad_idempotent,
    get_device,
    to_device,
    # Validation
    validate_bicomplex_tensor,
    print_bicomplex_info,
)

__all__ = [
    # === Activations ===
    "bicomplex_relu",
    "bicomplex_leaky_relu",
    "bicomplex_elu",
    "bicomplex_selu",
    "bicomplex_gelu",
    "bicomplex_sigmoid",
    "bicomplex_tanh",
    "bicomplex_softplus",
    "bicomplex_softsign",
    "bicomplex_hardtanh",
    "bicomplex_softmax",
    "bicomplex_log_softmax",
    "bicomplex_swish",
    "bicomplex_mish",
    "bicomplex_prelu",
    "bicomplex_rrelu",
    "bicomplex_threshold",
    "bicomplex_modulus_activation",
    "bicomplex_normalize",
    "bicomplex_phase_activation",
    "bicomplex_exp_activation",
    "bicomplex_bounded_exp_activation",
    "bicomplex_sin_activation",
    "bicomplex_cos_activation",
    "bicomplex_sinh_activation",
    "bicomplex_cosh_activation",
    "bicomplex_tanh_holomorphic",
    "bicomplex_polynomial_activation",
    "bicomplex_learnable_polynomial_activation",
    "bicomplex_rational_activation",
    "bicomplex_split_activation",
    # === Linear ===
    "bicomplex_linear",
    "bicomplex_bilinear",
    "bicomplex_matmul",
    "bicomplex_addmm",
    "bicomplex_addmv",
    "bicomplex_addr",
    # === Convolution ===
    "bicomplex_conv1d",
    "bicomplex_conv2d",
    "bicomplex_conv3d",
    "bicomplex_conv_transpose1d",
    "bicomplex_conv_transpose2d",
    "bicomplex_conv_transpose3d",
    "bicomplex_unfold",
    "bicomplex_fold",
    # === Pooling ===
    "bicomplex_max_pool1d",
    "bicomplex_max_pool2d",
    "bicomplex_max_pool3d",
    "bicomplex_avg_pool1d",
    "bicomplex_avg_pool2d",
    "bicomplex_avg_pool3d",
    "bicomplex_adaptive_max_pool1d",
    "bicomplex_adaptive_max_pool2d",
    "bicomplex_adaptive_max_pool3d",
    "bicomplex_adaptive_avg_pool1d",
    "bicomplex_adaptive_avg_pool2d",
    "bicomplex_adaptive_avg_pool3d",
    "bicomplex_lp_pool1d",
    "bicomplex_lp_pool2d",
    "bicomplex_max_unpool1d",
    "bicomplex_max_unpool2d",
    "bicomplex_max_unpool3d",
    "bicomplex_fractional_max_pool2d",
    "bicomplex_fractional_max_pool3d",
    # === Normalization ===
    "bicomplex_batch_norm",
    "bicomplex_layer_norm",
    "bicomplex_instance_norm",
    "bicomplex_group_norm",
    "bicomplex_local_response_norm",
    "bicomplex_lp_normalize",
    "bicomplex_rms_norm",
    "bicomplex_weight_norm",
    "bicomplex_spectral_norm",
    # === Dropout ===
    "bicomplex_dropout",
    "bicomplex_dropout1d",
    "bicomplex_dropout2d",
    "bicomplex_dropout3d",
    "bicomplex_alpha_dropout",
    "bicomplex_feature_alpha_dropout",
    "bicomplex_component_dropout",
    "bicomplex_zoneout",
    "bicomplex_dropconnect",
    "bicomplex_variational_dropout",
    "bicomplex_spatial_dropout",
    "bicomplex_gaussian_dropout",
    "bicomplex_concrete_dropout",
    # === Loss ===
    "mse_loss",
    "l1_loss",
    "smooth_l1_loss",
    "component_mse_loss",
    "cosine_embedding_loss",
    "triplet_margin_loss",
    "hinge_embedding_loss",
    "kl_divergence_loss",
    "margin_ranking_loss",
    "cauchy_loss",
    # === Attention ===
    "scaled_dot_product_attention",
    "multi_head_attention",
    "additive_attention",
    "self_attention",
    "cross_attention",
    "multiplicative_attention",
    "local_attention",
    # === Utilities ===
    "zeros_idempotent",
    "ones_idempotent",
    "randn_idempotent",
    "rand_idempotent",
    "randn_standard",
    "xavier_uniform_idempotent",
    "xavier_normal_idempotent",
    "kaiming_uniform_idempotent",
    "kaiming_normal_idempotent",
    "uniform_idempotent",
    "normal_idempotent",
    "real_to_bicomplex_idempotent",
    "complex_to_bicomplex_idempotent",
    "tensor_to_bicomplex",
    "extract_components_standard",
    "extract_components_idempotent",
    "compute_statistics_idempotent",
    "clip_grad_norm_idempotent",
    "zero_grad_idempotent",
    "get_device",
    "to_device",
    "validate_bicomplex_tensor",
    "print_bicomplex_info",
]
