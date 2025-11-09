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
from ..core.tensor_ops import matmul


def bicomplex_dropout(
        input: tuple[torch.Tensor, torch.Tensor],
        p: float = 0.5,
        training: bool = True,
        inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies dropout to bicomplex input in idempotent form.

    During training, randomly zeros elements with probability p and scales
    remaining elements by 1/(1-p). The same dropout mask is applied to both
    idempotent components to maintain bicomplex structure.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: any shape for each component
        p: Probability of an element being zeroed (default: 0.5)
        training: If True, apply dropout; if False, return input unchanged
        inplace: If True, modifies input in-place

    Returns:
        Bicomplex output tensor in idempotent form with dropout applied
        Shape: same as input

    Example:
        >>> input = (torch.randn(32, 128), torch.randn(32, 128))
        >>> output = bicomplex_dropout(input, p=0.5, training=True)
        >>> output[0].shape, output[1].shape
        (torch.Size([32, 128]), torch.Size([32, 128]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if not training or p == 0.0:
        return input

    if p == 1.0:
        return (torch.zeros_like(input[0]), torch.zeros_like(input[1]))

    # Generate a single mask and apply to both components to maintain structure
    mask = torch.empty_like(input[0]).bernoulli_(1 - p)
    scale = 1.0 / (1 - p)

    if inplace:
        input[0].mul_(mask).mul_(scale)
        input[1].mul_(mask).mul_(scale)
        return input
    else:
        output_e1 = input[0] * mask * scale
        output_e2 = input[1] * mask * scale
        return (output_e1, output_e2)


def bicomplex_dropout1d(
        input: tuple[torch.Tensor, torch.Tensor],
        p: float = 0.5,
        training: bool = True,
        inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 1D dropout to bicomplex input in idempotent form.

    Randomly zeros entire 1D feature maps (channels) with probability p.
    Useful for 1D convolutional layers.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, length) for each component
        p: Probability of a channel being zeroed (default: 0.5)
        training: If True, apply dropout; if False, return input unchanged
        inplace: If True, modifies input in-place

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: same as input

    Example:
        >>> input = (torch.randn(8, 64, 100), torch.randn(8, 64, 100))
        >>> output = bicomplex_dropout1d(input, p=0.2, training=True)
        >>> output[0].shape, output[1].shape
        (torch.Size([8, 64, 100]), torch.Size([8, 64, 100]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if not training or p == 0.0:
        return input

    # Apply feature dropout (dropout entire channels)
    output_e1 = torch.nn.functional.dropout1d(input[0], p, training, inplace)
    output_e2 = torch.nn.functional.dropout1d(input[1], p, training, inplace)

    return (output_e1, output_e2)


def bicomplex_dropout2d(
        input: tuple[torch.Tensor, torch.Tensor],
        p: float = 0.5,
        training: bool = True,
        inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 2D dropout to bicomplex input in idempotent form.

    Randomly zeros entire 2D feature maps (channels) with probability p.
    Useful for 2D convolutional layers in CNNs.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, height, width) for each component
        p: Probability of a channel being zeroed (default: 0.5)
        training: If True, apply dropout; if False, return input unchanged
        inplace: If True, modifies input in-place

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: same as input

    Example:
        >>> input = (torch.randn(8, 64, 28, 28), torch.randn(8, 64, 28, 28))
        >>> output = bicomplex_dropout2d(input, p=0.2, training=True)
        >>> output[0].shape, output[1].shape
        (torch.Size([8, 64, 28, 28]), torch.Size([8, 64, 28, 28]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if not training or p == 0.0:
        return input

    # Apply feature dropout (dropout entire channels)
    output_e1 = torch.nn.functional.dropout2d(input[0], p, training, inplace)
    output_e2 = torch.nn.functional.dropout2d(input[1], p, training, inplace)

    return (output_e1, output_e2)


def bicomplex_dropout3d(
        input: tuple[torch.Tensor, torch.Tensor],
        p: float = 0.5,
        training: bool = True,
        inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 3D dropout to bicomplex input in idempotent form.

    Randomly zeros entire 3D feature maps (channels) with probability p.
    Useful for 3D convolutional layers in volumetric CNNs.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, depth, height, width) for each component
        p: Probability of a channel being zeroed (default: 0.5)
        training: If True, apply dropout; if False, return input unchanged
        inplace: If True, modifies input in-place

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: same as input
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if not training or p == 0.0:
        return input

    # Apply feature dropout (dropout entire channels)
    output_e1 = torch.nn.functional.dropout3d(input[0], p, training, inplace)
    output_e2 = torch.nn.functional.dropout3d(input[1], p, training, inplace)

    return (output_e1, output_e2)


def bicomplex_alpha_dropout(
        input: tuple[torch.Tensor, torch.Tensor],
        p: float = 0.5,
        training: bool = True,
        inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies alpha dropout to bicomplex input in idempotent form.

    Alpha dropout maintains self-normalizing properties for SELU activation.
    It randomly sets inputs to the negative saturation value instead of zero,
    preserving mean and variance.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: any shape for each component
        p: Probability of an element being dropped (default: 0.5)
        training: If True, apply dropout; if False, return input unchanged
        inplace: If True, modifies input in-place

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: same as input

    Example:
        >>> input = (torch.randn(32, 128), torch.randn(32, 128))
        >>> output = bicomplex_alpha_dropout(input, p=0.1, training=True)
        >>> output[0].shape, output[1].shape
        (torch.Size([32, 128]), torch.Size([32, 128]))
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if not training or p == 0.0:
        return input

    # Apply alpha dropout component-wise
    output_e1 = torch.nn.functional.alpha_dropout(input[0], p, training, inplace)
    output_e2 = torch.nn.functional.alpha_dropout(input[1], p, training, inplace)

    return (output_e1, output_e2)


def bicomplex_feature_alpha_dropout(
        input: tuple[torch.Tensor, torch.Tensor],
        p: float = 0.5,
        training: bool = True,
        inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies feature-wise alpha dropout to bicomplex input.

    Similar to alpha_dropout but drops entire feature maps instead of
    individual elements. Combines alpha dropout with feature dropout.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, *) for each component
        p: Probability of a feature being dropped (default: 0.5)
        training: If True, apply dropout; if False, return input unchanged
        inplace: If True, modifies input in-place

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: same as input
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if not training or p == 0.0:
        return input

    # Apply feature alpha dropout component-wise
    output_e1 = torch.nn.functional.feature_alpha_dropout(input[0], p, training, inplace)
    output_e2 = torch.nn.functional.feature_alpha_dropout(input[1], p, training, inplace)

    return (output_e1, output_e2)


def bicomplex_component_dropout(
        input: tuple[torch.Tensor, torch.Tensor],
        p: float = 0.5,
        training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies dropout independently to each idempotent component.

    Unlike standard bicomplex_dropout which uses the same mask for both
    components, this applies independent masks to e1 and e2. This can
    provide stronger regularization but may break some bicomplex properties.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: any shape for each component
        p: Probability of an element being zeroed (default: 0.5)
        training: If True, apply dropout; if False, return input unchanged

    Returns:
        Bicomplex output tensor in idempotent form with independent dropout
        Shape: same as input

    Example:
        >>> input = (torch.randn(32, 128), torch.randn(32, 128))
        >>> output = bicomplex_component_dropout(input, p=0.3, training=True)
        >>> # e1 and e2 have different dropout masks
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if not training or p == 0.0:
        return input

    # Apply independent dropout to each component
    output_e1 = torch.nn.functional.dropout(input[0], p, training, inplace=False)
    output_e2 = torch.nn.functional.dropout(input[1], p, training, inplace=False)

    return (output_e1, output_e2)


def bicomplex_zoneout(
        input: tuple[torch.Tensor, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor],
        p: float = 0.5,
        training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies zoneout regularization to bicomplex recurrent networks.

    Zoneout randomly preserves hidden states from the previous timestep
    instead of updating them. It's a form of dropout for RNNs that maintains
    information flow through time.

    Args:
        input: New bicomplex hidden state in idempotent form
        hidden: Previous bicomplex hidden state in idempotent form
        p: Probability of preserving previous hidden state (default: 0.5)
        training: If True, apply zoneout; if False, interpolate

    Returns:
        Bicomplex output combining new and previous states

    Example:
        >>> new_hidden = (torch.randn(32, 256), torch.randn(32, 256))
        >>> prev_hidden = (torch.randn(32, 256), torch.randn(32, 256))
        >>> output = bicomplex_zoneout(new_hidden, prev_hidden, p=0.15, training=True)
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(hidden):
        raise ValueError("Hidden must be a bicomplex tensor in idempotent form")

    if not training:
        # During inference, use weighted average
        output_e1 = (1 - p) * input[0] + p * hidden[0]
        output_e2 = (1 - p) * input[1] + p * hidden[1]
        return (output_e1, output_e2)

    # During training, randomly choose between new and old state
    mask = torch.empty_like(input[0]).bernoulli_(1 - p)

    output_e1 = mask * input[0] + (1 - mask) * hidden[0]
    output_e2 = mask * input[1] + (1 - mask) * hidden[1]

    return (output_e1, output_e2)


def bicomplex_dropconnect(
        input: tuple[torch.Tensor, torch.Tensor],
        weight: tuple[torch.Tensor, torch.Tensor],
        bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        p: float = 0.5,
        training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies dropconnect to bicomplex linear layer.

    Instead of dropping activations (like dropout), dropconnect drops
    connections (weights) during training. This can provide better
    regularization for fully connected layers.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (*, in_features) for each component
        weight: Bicomplex weight matrix in idempotent form
                Shape: (out_features, in_features) for each component
        bias: Optional bicomplex bias in idempotent form
              Shape: (out_features,) for each component
        p: Probability of dropping a connection (default: 0.5)
        training: If True, apply dropconnect; if False, use full weights

    Returns:
        Bicomplex output tensor in idempotent form
        Shape: (*, out_features) for each component

    Example:
        >>> input = (torch.randn(32, 128), torch.randn(32, 128))
        >>> weight = (torch.randn(64, 128), torch.randn(64, 128))
        >>> bias = (torch.randn(64), torch.randn(64))
        >>> output = bicomplex_dropconnect(input, weight, bias, p=0.5, training=True)
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(weight):
        raise ValueError("Weight must be a bicomplex tensor in idempotent form")
    if bias is not None and not is_idempotent(bias):
        raise ValueError("Bias must be a bicomplex tensor in idempotent form")

    if not training or p == 0.0:
        # Standard linear operation
        output = matmul(input, (weight[0].t(), weight[1].t()))
        if bias is not None:
            output = (output[0] + bias[0], output[1] + bias[1])
        return output

    # Apply dropout to weights
    scale = 1.0 / (1 - p)
    mask_e1 = torch.empty_like(weight[0]).bernoulli_(1 - p) * scale
    mask_e2 = torch.empty_like(weight[1]).bernoulli_(1 - p) * scale

    dropped_weight = (weight[0] * mask_e1, weight[1] * mask_e2)

    # Apply linear transformation with dropped weights
    output = matmul(input, (dropped_weight[0].t(), dropped_weight[1].t()))

    if bias is not None:
        output = (output[0] + bias[0], output[1] + bias[1])

    return output


def bicomplex_variational_dropout(
        input: tuple[torch.Tensor, torch.Tensor],
        log_sigma: tuple[torch.Tensor, torch.Tensor],
        training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies variational dropout to bicomplex input.

    Variational dropout learns the dropout rate per weight/activation by
    treating it as a variational inference problem. Each element has its
    own dropout probability parameterized by log_sigma.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: any shape for each component
        log_sigma: Log standard deviation of dropout distribution
                   Shape: same as input for each component
        training: If True, sample dropout; if False, return expected value

    Returns:
        Bicomplex output tensor with variational dropout applied

    Example:
        >>> input = (torch.randn(32, 128), torch.randn(32, 128))
        >>> log_sigma = (torch.randn(32, 128) * 0.1, torch.randn(32, 128) * 0.1)
        >>> output = bicomplex_variational_dropout(input, log_sigma, training=True)
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(log_sigma):
        raise ValueError("log_sigma must be a bicomplex tensor in idempotent form")

    if not training:
        # During inference, use expected value (no dropout)
        return input

    # Sample noise from standard normal
    epsilon_e1 = torch.randn_like(input[0])
    epsilon_e2 = torch.randn_like(input[1])

    # Apply learned dropout: output = input * (1 + sigma * epsilon)
    sigma_e1 = torch.exp(log_sigma[0])
    sigma_e2 = torch.exp(log_sigma[1])

    output_e1 = input[0] * (1 + sigma_e1 * epsilon_e1)
    output_e2 = input[1] * (1 + sigma_e2 * epsilon_e2)

    return (output_e1, output_e2)


def bicomplex_spatial_dropout(
        input: tuple[torch.Tensor, torch.Tensor],
        p: float = 0.5,
        training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies spatial dropout to bicomplex input.

    Spatial dropout drops entire feature maps across all spatial locations.
    This is particularly useful for convolutional layers where nearby
    activations are strongly correlated.

    Alias for bicomplex_dropout2d for clarity in spatial contexts.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: (batch, channels, height, width) for each component
        p: Probability of dropping a channel (default: 0.5)
        training: If True, apply dropout; if False, return input unchanged

    Returns:
        Bicomplex output tensor with spatial dropout applied
    """
    return bicomplex_dropout2d(input, p, training, inplace=False)


def bicomplex_gaussian_dropout(
        input: tuple[torch.Tensor, torch.Tensor],
        p: float = 0.5,
        training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Gaussian dropout to bicomplex input.

    Instead of binary masks, Gaussian dropout multiplies inputs by noise
    sampled from N(1, sqrt(p/(1-p))). This provides a continuous relaxation
    of binary dropout.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: any shape for each component
        p: Dropout rate parameter (not direct probability)
        training: If True, apply dropout; if False, return input unchanged

    Returns:
        Bicomplex output tensor with Gaussian noise applied

    Example:
        >>> input = (torch.randn(32, 128), torch.randn(32, 128))
        >>> output = bicomplex_gaussian_dropout(input, p=0.2, training=True)
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    if not training or p == 0.0:
        return input

    if p >= 1.0:
        raise ValueError("p must be less than 1.0 for Gaussian dropout")

    # Standard deviation of Gaussian noise
    std = torch.sqrt(torch.tensor(p / (1.0 - p)))

    # Sample Gaussian noise centered at 1
    noise_e1 = torch.randn_like(input[0]) * std + 1.0
    noise_e2 = torch.randn_like(input[1]) * std + 1.0

    output_e1 = input[0] * noise_e1
    output_e2 = input[1] * noise_e2

    return (output_e1, output_e2)


def bicomplex_concrete_dropout(
        input: tuple[torch.Tensor, torch.Tensor],
        p_logit: tuple[torch.Tensor, torch.Tensor],
        temperature: float = 0.1,
        training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies concrete (continuous) dropout to bicomplex input.

    Concrete dropout provides a continuous, differentiable relaxation of
    discrete dropout masks, allowing the dropout probability to be learned
    via gradient descent.

    Args:
        input: Bicomplex input tensor in idempotent form
               Shape: any shape for each component
        p_logit: Logit of dropout probability (learnable parameter)
                 Shape: scalar or broadcastable for each component
        temperature: Temperature for Gumbel-Softmax (lower = closer to discrete)
        training: If True, sample masks; if False, use expected value

    Returns:
        Bicomplex output with concrete dropout applied

    Example:
        >>> input = (torch.randn(32, 128), torch.randn(32, 128))
        >>> p_logit = (torch.tensor(0.0), torch.tensor(0.0))  # learnable
        >>> output = bicomplex_concrete_dropout(input, p_logit, temperature=0.1)
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(p_logit):
        raise ValueError("p_logit must be a bicomplex tensor in idempotent form")

    # Convert logit to probability
    p_e1 = torch.sigmoid(p_logit[0])
    p_e2 = torch.sigmoid(p_logit[1])

    if not training:
        # During inference, use expected value
        output_e1 = input[0] * (1 - p_e1)
        output_e2 = input[1] * (1 - p_e2)
        return (output_e1, output_e2)

    # Sample from Gumbel distribution
    uniform_e1 = torch.rand_like(input[0])
    uniform_e2 = torch.rand_like(input[1])

    gumbel_e1 = -torch.log(-torch.log(uniform_e1 + 1e-8) + 1e-8)
    gumbel_e2 = -torch.log(-torch.log(uniform_e2 + 1e-8) + 1e-8)

    # Concrete distribution (Gumbel-Softmax trick)
    concrete_e1 = torch.sigmoid((torch.log(p_e1 + 1e-8) - torch.log(1 - p_e1 + 1e-8) + gumbel_e1) / temperature)
    concrete_e2 = torch.sigmoid((torch.log(p_e2 + 1e-8) - torch.log(1 - p_e2 + 1e-8) + gumbel_e2) / temperature)

    # Apply dropout mask
    output_e1 = input[0] * (1 - concrete_e1) / (1 - p_e1 + 1e-8)
    output_e2 = input[1] * (1 - concrete_e2) / (1 - p_e2 + 1e-8)

    return (output_e1, output_e2)