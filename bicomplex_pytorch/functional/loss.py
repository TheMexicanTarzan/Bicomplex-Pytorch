"""
Loss functions for bicomplex neural networks.

All loss functions operate on bicomplex tensors in idempotent form (e1, e2).
"""

import torch
from typing import Optional, Literal

from ..core.arithmetic import (
    subtract_idempotent,
    modulus_squared,
    modulus,
    squared_idempotent_norm,
    multiply_idempotent,
    conjugate_idempotent,
)
from ..core.tensor_ops import mean, sum as bc_sum
from ..core.representations import is_idempotent

ReductionType = Literal['none', 'mean', 'sum']


def _apply_reduction(loss: torch.Tensor, reduction: ReductionType) -> torch.Tensor:
    """Apply reduction to loss tensor."""
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


def mse_loss(
        input: tuple[torch.Tensor, torch.Tensor],
        target: tuple[torch.Tensor, torch.Tensor],
        reduction: ReductionType = 'mean'
) -> torch.Tensor:
    """
    Mean Squared Error loss for bicomplex tensors.

    Computes ||input - target||² using the total modulus squared.

    Args:
        input: Predicted bicomplex tensor in idempotent form
        target: Target bicomplex tensor in idempotent form
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        Real-valued loss tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(target):
        raise ValueError("Target must be a bicomplex tensor in idempotent form")

    # Compute difference
    diff = subtract_idempotent(input, target)

    # Compute squared modulus: |e1|² + |e2|²
    squared_error = modulus_squared(diff)

    return _apply_reduction(squared_error, reduction)


def l1_loss(
        input: tuple[torch.Tensor, torch.Tensor],
        target: tuple[torch.Tensor, torch.Tensor],
        reduction: ReductionType = 'mean'
) -> torch.Tensor:
    """
    L1 (Mean Absolute Error) loss for bicomplex tensors.

    Computes ||input - target|| using the total modulus.

    Args:
        input: Predicted bicomplex tensor in idempotent form
        target: Target bicomplex tensor in idempotent form
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        Real-valued loss tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(target):
        raise ValueError("Target must be a bicomplex tensor in idempotent form")

    # Compute difference
    diff = subtract_idempotent(input, target)

    # Compute modulus: sqrt(|e1|² + |e2|²)
    absolute_error = modulus(diff)

    return _apply_reduction(absolute_error, reduction)


def smooth_l1_loss(
        input: tuple[torch.Tensor, torch.Tensor],
        target: tuple[torch.Tensor, torch.Tensor],
        reduction: ReductionType = 'mean',
        beta: float = 1.0
) -> torch.Tensor:
    """
    Smooth L1 loss (Huber loss) for bicomplex tensors.

    Uses L2 loss when error is small, L1 loss when error is large.

    Args:
        input: Predicted bicomplex tensor in idempotent form
        target: Target bicomplex tensor in idempotent form
        reduction: 'none' | 'mean' | 'sum'
        beta: Threshold for switching between L1 and L2

    Returns:
        Real-valued loss tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(target):
        raise ValueError("Target must be a bicomplex tensor in idempotent form")

    # Compute difference
    diff = subtract_idempotent(input, target)

    # Compute modulus
    abs_diff = modulus(diff)

    # Smooth L1: 0.5 * x² / beta  if |x| < beta
    #            |x| - 0.5 * beta  otherwise
    loss = torch.where(
        abs_diff < beta,
        0.5 * abs_diff ** 2 / beta,
        abs_diff - 0.5 * beta
    )

    return _apply_reduction(loss, reduction)


def component_mse_loss(
        input: tuple[torch.Tensor, torch.Tensor],
        target: tuple[torch.Tensor, torch.Tensor],
        reduction: ReductionType = 'mean',
        weight_e1: float = 1.0,
        weight_e2: float = 1.0
) -> torch.Tensor:
    """
    Component-wise MSE loss for bicomplex tensors.

    Computes weighted sum of |e1_pred - e1_target|² and |e2_pred - e2_target|².
    Useful when you want to penalize errors in each idempotent component differently.

    Args:
        input: Predicted bicomplex tensor in idempotent form
        target: Target bicomplex tensor in idempotent form
        reduction: 'none' | 'mean' | 'sum'
        weight_e1: Weight for e1 component loss
        weight_e2: Weight for e2 component loss

    Returns:
        Real-valued loss tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(target):
        raise ValueError("Target must be a bicomplex tensor in idempotent form")

    # Compute difference
    diff = subtract_idempotent(input, target)

    # Get component-wise squared norms
    e1_sq, e2_sq = squared_idempotent_norm(diff)

    # Weighted sum
    loss = weight_e1 * e1_sq + weight_e2 * e2_sq

    return _apply_reduction(loss, reduction)


def cosine_embedding_loss(
        input1: tuple[torch.Tensor, torch.Tensor],
        input2: tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        margin: float = 0.0,
        reduction: ReductionType = 'mean'
) -> torch.Tensor:
    """
    Cosine embedding loss for bicomplex tensors.

    Measures cosine similarity between bicomplex embeddings.

    Args:
        input1: First bicomplex tensor in idempotent form
        input2: Second bicomplex tensor in idempotent form
        target: Labels (1 for similar, -1 for dissimilar)
        margin: Margin for dissimilar pairs
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        Real-valued loss tensor
    """
    if not is_idempotent(input1):
        raise ValueError("Input1 must be a bicomplex tensor in idempotent form")
    if not is_idempotent(input2):
        raise ValueError("Input2 must be a bicomplex tensor in idempotent form")

    # Compute dot product: Re(input1* · input2)
    # For idempotent: (e1*, e2*) · (e1', e2') = (e2·e1', e1·e2')
    conjugate1 = conjugate_idempotent(input1)
    product = multiply_idempotent(conjugate1, input2)

    # Take real part (average of components for bicomplex)
    dot_product = 0.5 * (product[0].real + product[1].real)

    # Compute norms
    norm1 = modulus(input1)
    norm2 = modulus(input2)

    # Cosine similarity
    cos_sim = dot_product / (norm1 * norm2 + 1e-8)

    # Loss computation
    loss = torch.where(
        target == 1,
        1 - cos_sim,
        torch.clamp(cos_sim - margin, min=0.0)
    )

    return _apply_reduction(loss, reduction)


def triplet_margin_loss(
        anchor: tuple[torch.Tensor, torch.Tensor],
        positive: tuple[torch.Tensor, torch.Tensor],
        negative: tuple[torch.Tensor, torch.Tensor],
        margin: float = 1.0,
        p: float = 2.0,
        reduction: ReductionType = 'mean'
) -> torch.Tensor:
    """
    Triplet margin loss for bicomplex tensors.

    Ensures anchor is closer to positive than to negative by a margin.

    Args:
        anchor: Anchor bicomplex tensor in idempotent form
        positive: Positive bicomplex tensor in idempotent form
        negative: Negative bicomplex tensor in idempotent form
        margin: Minimum distance margin
        p: Norm degree (2 for Euclidean distance)
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        Real-valued loss tensor
    """
    if not is_idempotent(anchor):
        raise ValueError("Anchor must be a bicomplex tensor in idempotent form")
    if not is_idempotent(positive):
        raise ValueError("Positive must be a bicomplex tensor in idempotent form")
    if not is_idempotent(negative):
        raise ValueError("Negative must be a bicomplex tensor in idempotent form")

    # Compute distances
    diff_pos = subtract_idempotent(anchor, positive)
    diff_neg = subtract_idempotent(anchor, negative)

    if p == 2.0:
        dist_pos = modulus(diff_pos)
        dist_neg = modulus(diff_neg)
    elif p == 1.0:
        dist_pos = modulus(diff_pos)
        dist_neg = modulus(diff_neg)
    else:
        # General p-norm on modulus
        dist_pos = modulus(diff_pos) ** p
        dist_neg = modulus(diff_neg) ** p

    # Triplet loss
    loss = torch.clamp(dist_pos - dist_neg + margin, min=0.0)

    return _apply_reduction(loss, reduction)


def hinge_embedding_loss(
        input: tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        margin: float = 1.0,
        reduction: ReductionType = 'mean'
) -> torch.Tensor:
    """
    Hinge embedding loss for bicomplex tensors.

    Args:
        input: Bicomplex tensor in idempotent form
        target: Labels (1 or -1)
        margin: Margin for negative samples
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        Real-valued loss tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    # Compute modulus
    mod = modulus(input)

    # Hinge loss
    loss = torch.where(
        target == 1,
        mod,
        torch.clamp(margin - mod, min=0.0)
    )

    return _apply_reduction(loss, reduction)


def kl_divergence_loss(
        input: tuple[torch.Tensor, torch.Tensor],
        target: tuple[torch.Tensor, torch.Tensor],
        reduction: ReductionType = 'mean'
) -> torch.Tensor:
    """
    KL divergence loss for bicomplex probability distributions.

    Treats |e1|² and |e2|² as probability-like quantities and computes
    KL divergence on each component.

    Args:
        input: Predicted bicomplex tensor (log probabilities) in idempotent form
        target: Target bicomplex tensor (probabilities) in idempotent form
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        Real-valued loss tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(target):
        raise ValueError("Target must be a bicomplex tensor in idempotent form")

    # Compute component-wise KL divergence
    # KL(P||Q) = sum(P * log(P/Q))
    input_e1 = torch.abs(input[0])
    input_e2 = torch.abs(input[1])
    target_e1 = torch.abs(target[0])
    target_e2 = torch.abs(target[1])

    kl_e1 = torch.nn.functional.kl_div(
        torch.log(input_e1 + 1e-8),
        target_e1,
        reduction='none',
        log_target=False
    )
    kl_e2 = torch.nn.functional.kl_div(
        torch.log(input_e2 + 1e-8),
        target_e2,
        reduction='none',
        log_target=False
    )

    loss = kl_e1 + kl_e2

    return _apply_reduction(loss, reduction)


def margin_ranking_loss(
        input1: tuple[torch.Tensor, torch.Tensor],
        input2: tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        margin: float = 0.0,
        reduction: ReductionType = 'mean'
) -> torch.Tensor:
    """
    Margin ranking loss for bicomplex tensors.

    Args:
        input1: First bicomplex tensor in idempotent form
        input2: Second bicomplex tensor in idempotent form
        target: Labels (1: input1 > input2, -1: input2 > input1)
        margin: Ranking margin
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        Real-valued loss tensor
    """
    if not is_idempotent(input1):
        raise ValueError("Input1 must be a bicomplex tensor in idempotent form")
    if not is_idempotent(input2):
        raise ValueError("Input2 must be a bicomplex tensor in idempotent form")

    # Use modulus as the ranking score
    score1 = modulus(input1)
    score2 = modulus(input2)

    # Margin ranking loss
    loss = torch.clamp(-target * (score1 - score2) + margin, min=0.0)

    return _apply_reduction(loss, reduction)


def cauchy_loss(
        input: tuple[torch.Tensor, torch.Tensor],
        target: tuple[torch.Tensor, torch.Tensor],
        reduction: ReductionType = 'mean',
        c: float = 1.0
) -> torch.Tensor:
    """
    Cauchy loss (robust loss function) for bicomplex tensors.

    More robust to outliers than MSE.
    Loss = c² * log(1 + (||x||/c)²)

    Args:
        input: Predicted bicomplex tensor in idempotent form
        target: Target bicomplex tensor in idempotent form
        reduction: 'none' | 'mean' | 'sum'
        c: Scale parameter

    Returns:
        Real-valued loss tensor
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(target):
        raise ValueError("Target must be a bicomplex tensor in idempotent form")

    # Compute difference
    diff = subtract_idempotent(input, target)

    # Compute modulus
    mod = modulus(diff)

    # Cauchy loss
    loss = c ** 2 * torch.log(1 + (mod / c) ** 2)

    return _apply_reduction(loss, reduction)