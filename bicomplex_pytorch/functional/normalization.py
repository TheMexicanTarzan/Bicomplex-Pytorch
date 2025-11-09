import torch
from typing import Optional, Literal

from ..core.arithmetic import (
    exp_idempotent,
    log_idempotent,
    modulus,
    modulus_squared,
    divide_idempotent,
    add_idempotent,
    scalar_multiply_idempotent,
    multiply_idempotent,
    sqrt_idempotent,
    power_idempotent,
)
from ..core.representations import is_idempotent


def bicomplex_batch_norm(
        input: tuple[torch.Tensor, torch.Tensor],
        running_mean: Optional[tuple[torch.Tensor, torch.Tensor]],
        running_var: Optional[tuple[torch.Tensor, torch.Tensor]],
        weight: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies batch normalization to bicomplex input in idempotent form.

    Batch normalization is applied component-wise to each idempotent component.
    Each component is normalized independently using its own running statistics.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, *) for each component
        running_mean: Running mean for each component
                      Shape: (channels,) for each component
        running_var: Running variance for each component
                     Shape: (channels,) for each component
        weight: Optional learnable scale parameter (gamma)
                Shape: (channels,) for each component
        bias: Optional learnable shift parameter (beta)
              Shape: (channels,) for each component
        training: If True, use batch statistics; if False, use running statistics
        momentum: Momentum for running statistics update (default: 0.1)
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized bicomplex tensor in idempotent form
        Shape: same as input

    Example:
        >>> input = (torch.randn(32, 64, 28, 28), torch.randn(32, 64, 28, 28))
        >>> running_mean = (torch.zeros(64), torch.zeros(64))
        >>> running_var = (torch.ones(64), torch.ones(64))
        >>> weight = (torch.ones(64), torch.ones(64))
        >>> bias = (torch.zeros(64), torch.zeros(64))
        >>> output = bicomplex_batch_norm(input, running_mean, running_var,
        ...                               weight, bias, training=True)
        >>> output[0].shape, output[1].shape
        (torch.Size([32, 64, 28, 28]), torch.Size([32, 64, 28, 28]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if running_mean is not None and not is_idempotent(running_mean):
        raise ValueError("Running mean must be a bicomplex tensor in idempotent form")
    if running_var is not None and not is_idempotent(running_var):
        raise ValueError("Running variance must be a bicomplex tensor in idempotent form")
    if weight is not None and not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    # Apply batch normalization to e1 component
    output_e1 = torch.nn.functional.batch_norm(
        input[0],
        running_mean[0] if running_mean is not None else None,
        running_var[0] if running_var is not None else None,
        weight[0] if weight is not None else None,
        bias[0] if bias is not None else None,
        training,
        momentum,
        eps
    )

    # Apply batch normalization to e2 component
    output_e2 = torch.nn.functional.batch_norm(
        input[1],
        running_mean[1] if running_mean is not None else None,
        running_var[1] if running_var is not None else None,
        weight[1] if weight is not None else None,
        bias[1] if bias is not None else None,
        training,
        momentum,
        eps
    )

    return (output_e1, output_e2)


def bicomplex_layer_norm(
        input: tuple[torch.Tensor, torch.Tensor],
        normalized_shape: Union[int, tuple[int, ...]],
        weight: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies layer normalization to bicomplex input in idempotent form.

    Layer normalization is applied component-wise to each idempotent component.
    Normalizes over the last D dimensions where D is len(normalized_shape).

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (*, normalized_shape) for each component
        normalized_shape: Input shape from an expected input of size
                          [*, normalized_shape[0], normalized_shape[1], ...]
                          Can be int for 1D or tuple for multi-dimensional
        weight: Optional learnable scale parameter (gamma)
                Shape: normalized_shape for each component
        bias: Optional learnable shift parameter (beta)
              Shape: normalized_shape for each component
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized bicomplex tensor in idempotent form
        Shape: same as input

    Example:
        >>> # Normalize over last dimension
        >>> input = (torch.randn(32, 10, 512), torch.randn(32, 10, 512))
        >>> weight = (torch.ones(512), torch.ones(512))
        >>> bias = (torch.zeros(512), torch.zeros(512))
        >>> output = bicomplex_layer_norm(input, 512, weight, bias)
        >>> output[0].shape, output[1].shape
        (torch.Size([32, 10, 512]), torch.Size([32, 10, 512]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if weight is not None and not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    # Convert normalized_shape to tuple if it's an int
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    # Apply layer normalization to e1 component
    output_e1 = torch.nn.functional.layer_norm(
        input[0],
        normalized_shape,
        weight[0] if weight is not None else None,
        bias[0] if bias is not None else None,
        eps
    )

    # Apply layer normalization to e2 component
    output_e2 = torch.nn.functional.layer_norm(
        input[1],
        normalized_shape,
        weight[1] if weight is not None else None,
        bias[1] if bias is not None else None,
        eps
    )

    return (output_e1, output_e2)


def bicomplex_instance_norm(
        input: tuple[torch.Tensor, torch.Tensor],
        running_mean: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        running_var: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        weight: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies instance normalization to bicomplex input in idempotent form.

    Instance normalization normalizes across spatial dimensions for each
    channel and batch element independently. Commonly used in style transfer.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, *) for each component
        running_mean: Running mean for each component (optional)
                      Shape: (channels,) for each component
        running_var: Running variance for each component (optional)
                     Shape: (channels,) for each component
        weight: Optional learnable scale parameter
                Shape: (channels,) for each component
        bias: Optional learnable shift parameter
              Shape: (channels,) for each component
        use_input_stats: If True, use input statistics; if False, use running statistics
        momentum: Momentum for running statistics update (default: 0.1)
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized bicomplex tensor in idempotent form
        Shape: same as input

    Example:
        >>> input = (torch.randn(8, 64, 32, 32), torch.randn(8, 64, 32, 32))
        >>> output = bicomplex_instance_norm(input)
        >>> output[0].shape, output[1].shape
        (torch.Size([8, 64, 32, 32]), torch.Size([8, 64, 32, 32]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if running_mean is not None and not is_idempotent(running_mean):
        raise ValueError("Running mean must be a bicomplex tensor in idempotent form")
    if running_var is not None and not is_idempotent(running_var):
        raise ValueError("Running variance must be a bicomplex tensor in idempotent form")
    if weight is not None and not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    # Apply instance normalization to e1 component
    output_e1 = torch.nn.functional.instance_norm(
        input[0],
        running_mean[0] if running_mean is not None else None,
        running_var[0] if running_var is not None else None,
        weight[0] if weight is not None else None,
        bias[0] if bias is not None else None,
        use_input_stats,
        momentum,
        eps
    )

    # Apply instance normalization to e2 component
    output_e2 = torch.nn.functional.instance_norm(
        input[1],
        running_mean[1] if running_mean is not None else None,
        running_var[1] if running_var is not None else None,
        weight[1] if weight is not None else None,
        bias[1] if bias is not None else None,
        use_input_stats,
        momentum,
        eps
    )

    return (output_e1, output_e2)


def bicomplex_group_norm(
        input: tuple[torch.Tensor, torch.Tensor],
        num_groups: int,
        weight: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies group normalization to bicomplex input in idempotent form.

    Group normalization divides channels into groups and normalizes within each group.
    It's a middle ground between layer norm (num_groups=1) and instance norm
    (num_groups=num_channels).

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, *) for each component
        num_groups: Number of groups to divide channels into
                    Must divide num_channels evenly
        weight: Optional learnable scale parameter
                Shape: (channels,) for each component
        bias: Optional learnable shift parameter
              Shape: (channels,) for each component
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized bicomplex tensor in idempotent form
        Shape: same as input

    Example:
        >>> input = (torch.randn(8, 64, 32, 32), torch.randn(8, 64, 32, 32))
        >>> weight = (torch.ones(64), torch.ones(64))
        >>> bias = (torch.zeros(64), torch.zeros(64))
        >>> output = bicomplex_group_norm(input, num_groups=8, weight=weight, bias=bias)
        >>> output[0].shape, output[1].shape
        (torch.Size([8, 64, 32, 32]), torch.Size([8, 64, 32, 32]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if weight is not None and not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    # Apply group normalization to e1 component
    output_e1 = torch.nn.functional.group_norm(
        input[0],
        num_groups,
        weight[0] if weight is not None else None,
        bias[0] if bias is not None else None,
        eps
    )

    # Apply group normalization to e2 component
    output_e2 = torch.nn.functional.group_norm(
        input[1],
        num_groups,
        weight[1] if weight is not None else None,
        bias[1] if bias is not None else None,
        eps
    )

    return (output_e1, output_e2)


def bicomplex_local_response_norm(
        input: tuple[torch.Tensor, torch.Tensor],
        size: int,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies local response normalization to bicomplex input in idempotent form.

    LRN performs a kind of "lateral inhibition" by normalizing over local input
    regions. Used in older architectures like AlexNet.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, *) for each component
        size: Number of neighboring channels to use for normalization
        alpha: Multiplicative factor (default: 1e-4)
        beta: Exponent (default: 0.75)
        k: Additive factor (default: 1.0)

    Returns:
        Normalized bicomplex tensor in idempotent form
        Shape: same as input

    Example:
        >>> input = (torch.randn(8, 96, 55, 55), torch.randn(8, 96, 55, 55))
        >>> output = bicomplex_local_response_norm(input, size=5)
        >>> output[0].shape, output[1].shape
        (torch.Size([8, 96, 55, 55]), torch.Size([8, 96, 55, 55]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Apply local response normalization to e1 component
    output_e1 = torch.nn.functional.local_response_norm(
        input[0], size, alpha, beta, k
    )

    # Apply local response normalization to e2 component
    output_e2 = torch.nn.functional.local_response_norm(
        input[1], size, alpha, beta, k
    )

    return (output_e1, output_e2)


def bicomplex_normalize(
        input: tuple[torch.Tensor, torch.Tensor],
        p: float = 2.0,
        dim: int = 1,
        eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes bicomplex tensor using Lp norm.

    For each sample, normalizes along the specified dimension using the Lp norm.
    This is applied component-wise to maintain bicomplex structure.

    Args:
        input: Bicomplex input tensor in idempotent form
        p: Norm order (default: 2 for L2 norm)
        dim: Dimension along which to normalize (default: 1)
        eps: Small constant to avoid division by zero (default: 1e-12)

    Returns:
        Normalized bicomplex tensor in idempotent form

    Example:
        >>> # Normalize feature vectors
        >>> input = (torch.randn(32, 128), torch.randn(32, 128))
        >>> output = bicomplex_normalize(input, p=2, dim=1)
        >>> # Each row now has unit L2 norm (per component)
        >>> torch.allclose(output[0].norm(p=2, dim=1), torch.ones(32), atol=1e-6)
        True
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Apply normalization to e1 component
    output_e1 = torch.nn.functional.normalize(input[0], p=p, dim=dim, eps=eps)

    # Apply normalization to e2 component
    output_e2 = torch.nn.functional.normalize(input[1], p=p, dim=dim, eps=eps)

    return (output_e1, output_e2)


def bicomplex_rms_norm(
        input: tuple[torch.Tensor, torch.Tensor],
        normalized_shape: Union[int, tuple[int, ...]],
        weight: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Root Mean Square (RMS) normalization to bicomplex input.

    RMS normalization is a simpler alternative to layer normalization that
    only normalizes by the root mean square, without centering. Used in
    modern architectures like LLaMA.

    RMS(x) = sqrt(mean(x^2) + eps)
    Normalized output = x / RMS(x) * weight

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (*, normalized_shape) for each component
        normalized_shape: Shape over which to compute RMS
                          Can be int for 1D or tuple for multi-dimensional
        weight: Optional learnable scale parameter
                Shape: normalized_shape for each component
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized bicomplex tensor in idempotent form
        Shape: same as input

    Example:
        >>> input = (torch.randn(32, 128, 512), torch.randn(32, 128, 512))
        >>> weight = (torch.ones(512), torch.ones(512))
        >>> output = bicomplex_rms_norm(input, 512, weight)
        >>> output[0].shape, output[1].shape
        (torch.Size([32, 128, 512]), torch.Size([32, 128, 512]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if weight is not None and not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")

    # Convert normalized_shape to tuple if it's an int
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    # Compute dimensions to reduce over (last len(normalized_shape) dimensions)
    dims = tuple(range(-len(normalized_shape), 0))

    # Apply RMS normalization to e1 component
    rms_e1 = torch.sqrt(torch.mean(input[0] ** 2, dim=dims, keepdim=True) + eps)
    output_e1 = input[0] / rms_e1
    if weight is not None:
        output_e1 = output_e1 * weight[0]

    # Apply RMS normalization to e2 component
    rms_e2 = torch.sqrt(torch.mean(input[1] ** 2, dim=dims, keepdim=True) + eps)
    output_e2 = input[1] / rms_e2
    if weight is not None:
        output_e2 = output_e2 * weight[1]

    return (output_e1, output_e2)


def bicomplex_weight_norm(
        module_weight: tuple[torch.Tensor, torch.Tensor],
        dim: int = 0
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Applies weight normalization to bicomplex weights.

    Weight normalization reparameterizes weights as:
    w = g * v / ||v||
    where g is a scalar and v is the direction vector.

    This function computes the normalized weight and the norm (g).

    Args:
        module_weight: Bicomplex weight tensor in idempotent form
        dim: Dimension along which to compute the norm (default: 0)

    Returns:
        Tuple of (normalized_weight, weight_g) where:
        - normalized_weight: Weight normalized to unit norm along dim
        - weight_g: The norm values (for reparameterization)

    Example:
        >>> weight = (torch.randn(64, 32, 3, 3), torch.randn(64, 32, 3, 3))
        >>> normalized_weight, weight_g = bicomplex_weight_norm(weight, dim=0)
        >>> normalized_weight[0].shape, weight_g[0].shape
        (torch.Size([64, 32, 3, 3]), torch.Size([1, 32, 3, 3]))
    """
    if not is_idempotent(module_weight):
        raise ValueError("Module weight must be a bicomplex tensor in idempotent form")

    # Compute norm along specified dimension for e1
    norm_e1 = torch.norm(module_weight[0], p=2, dim=dim, keepdim=True)
    normalized_e1 = module_weight[0] / (norm_e1 + 1e-12)

    # Compute norm along specified dimension for e2
    norm_e2 = torch.norm(module_weight[1], p=2, dim=dim, keepdim=True)
    normalized_e2 = module_weight[1] / (norm_e2 + 1e-12)

    return ((normalized_e1, normalized_e2), (norm_e1, norm_e2))


def bicomplex_spectral_norm(
        weight: tuple[torch.Tensor, torch.Tensor],
        u: tuple[torch.Tensor, torch.Tensor],
        n_power_iterations: int = 1,
        eps: float = 1e-12
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Applies spectral normalization to bicomplex weights.

    Spectral normalization normalizes weights by their largest singular value,
    which helps stabilize GAN training. Uses power iteration to approximate
    the largest singular value.

    Args:
        weight: Bicomplex weight tensor in idempotent form
                Shape: (out_features, in_features) or (out_channels, in_channels, ...)
        u: Left singular vector approximation (updated in-place)
           Shape: (out_features,) for each component
        n_power_iterations: Number of power iterations (default: 1)
        eps: Small constant for numerical stability (default: 1e-12)

    Returns:
        Tuple of (normalized_weight, updated_u) where:
        - normalized_weight: Weight normalized by spectral norm
        - updated_u: Updated left singular vector

    Example:
        >>> weight = (torch.randn(64, 128), torch.randn(64, 128))
        >>> u = (torch.randn(64), torch.randn(64))
        >>> normalized_weight, u = bicomplex_spectral_norm(weight, u, n_power_iterations=1)
        >>> normalized_weight[0].shape
        torch.Size([64, 128])
    """
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if not is_idempotent(u):
        raise ValueError("u must be a bicomplex tensor in idempotent form")

    # Reshape weight for matrix operations if needed
    weight_mat_e1 = weight[0].reshape(weight[0].size(0), -1)
    weight_mat_e2 = weight[1].reshape(weight[1].size(0), -1)

    # Power iteration for e1
    u_e1 = u[0]
    for _ in range(n_power_iterations):
        v_e1 = torch.nn.functional.normalize(torch.mv(weight_mat_e1.t(), u_e1), dim=0, eps=eps)
        u_e1 = torch.nn.functional.normalize(torch.mv(weight_mat_e1, v_e1), dim=0, eps=eps)

    # Compute spectral norm for e1
    sigma_e1 = torch.dot(u_e1, torch.mv(weight_mat_e1, v_e1))
    normalized_weight_e1 = weight[0] / sigma_e1

    # Power iteration for e2
    u_e2 = u[1]
    for _ in range(n_power_iterations):
        v_e2 = torch.nn.functional.normalize(torch.mv(weight_mat_e2.t(), u_e2), dim=0, eps=eps)
        u_e2 = torch.nn.functional.normalize(torch.mv(weight_mat_e2, v_e2), dim=0, eps=eps)

    # Compute spectral norm for e2
    sigma_e2 = torch.dot(u_e2, torch.mv(weight_mat_e2, v_e2))
    normalized_weight_e2 = weight[1] / sigma_e2

    return ((normalized_weight_e1, normalized_weight_e2), (u_e1.detach(), u_e2.detach()))