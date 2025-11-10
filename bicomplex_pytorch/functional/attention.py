"""
Attention mechanisms for bicomplex neural networks.

All operations work on bicomplex tensors in idempotent form (e1, e2).
"""

import torch
import math
from typing import Optional, Tuple

from ..core.arithmetic import (
    multiply_idempotent,
    conjugate_idempotent,
    modulus,
    add_idempotent,
    scalar_multiply_idempotent,
)
from ..core.tensor_ops import (
    matmul,
    bmm,
    transpose,
    softmax as bc_softmax,
)
from ..core.representations import is_idempotent


def scaled_dot_product_attention(
        query: tuple[torch.Tensor, torch.Tensor],
        key: tuple[torch.Tensor, torch.Tensor],
        value: tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        training: bool = True
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Scaled dot-product attention for bicomplex tensors.

    Computes: Attention(Q, K, V) = softmax(QK^*/√d_k) V
    where K^* is the bicomplex conjugate of K.

    Args:
        query: Query tensor in idempotent form, shape (batch, seq_len, d_k)
        key: Key tensor in idempotent form, shape (batch, seq_len, d_k)
        value: Value tensor in idempotent form, shape (batch, seq_len, d_v)
        attn_mask: Optional mask tensor, shape (seq_len, seq_len) or (batch, seq_len, seq_len)
        dropout_p: Dropout probability for attention weights
        is_causal: Whether to apply causal masking
        training: Whether in training mode (affects dropout)

    Returns:
        Tuple of:
            - Output tensor in idempotent form, shape (batch, seq_len, d_v)
            - Attention weights, shape (batch, seq_len, seq_len)
    """
    if not is_idempotent(query):
        raise ValueError("Query must be a bicomplex tensor in idempotent form")
    if not is_idempotent(key):
        raise ValueError("Key must be a bicomplex tensor in idempotent form")
    if not is_idempotent(value):
        raise ValueError("Value must be a bicomplex tensor in idempotent form")

    # Get dimensions
    batch_size = query[0].shape[0]
    seq_len_q = query[0].shape[1]
    seq_len_k = key[0].shape[1]
    d_k = query[0].shape[-1]

    # Conjugate keys: (e1, e2)* = (e2, e1)
    key_conj = conjugate_idempotent(key)

    # Transpose key: (batch, seq_len, d_k) -> (batch, d_k, seq_len)
    key_conj_t = transpose(key_conj, -2, -1)

    # Compute QK^T: (batch, seq_len_q, d_k) @ (batch, d_k, seq_len_k)
    # Result: (batch, seq_len_q, seq_len_k)
    scores = matmul(query, key_conj_t)

    # Use modulus for attention scores (real-valued)
    # This gives a single real attention score per query-key pair
    attn_scores = modulus(scores)  # (batch, seq_len_q, seq_len_k)

    # Scale by sqrt(d_k)
    scaling_factor = math.sqrt(d_k if isinstance(d_k, (int, float)) else d_k.item())
    attn_scores = attn_scores / scaling_factor

    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=attn_scores.device),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    # Apply attention mask if provided
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        else:
            attn_scores = attn_scores + attn_mask

    # Softmax to get attention weights
    attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq_len_q, seq_len_k)

    # Apply dropout to attention weights
    if dropout_p > 0.0 and training:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=training)

    # Apply attention to values
    # We need to convert real attention weights to bicomplex form for multiplication
    # Use the same weight for both e1 and e2 components
    attn_weights_bc = (
        attn_weights.unsqueeze(-1).expand_as(value[0]),
        attn_weights.unsqueeze(-1).expand_as(value[1])
    )

    # Weighted sum of values: (batch, seq_len_q, seq_len_k) @ (batch, seq_len_k, d_v)
    # We'll do this component-wise
    output_e1 = torch.matmul(attn_weights, value[0])  # (batch, seq_len_q, d_v)
    output_e2 = torch.matmul(attn_weights, value[1])  # (batch, seq_len_q, d_v)

    output = (output_e1, output_e2)

    return output, attn_weights


def multi_head_attention(
        query: tuple[torch.Tensor, torch.Tensor],
        key: tuple[torch.Tensor, torch.Tensor],
        value: tuple[torch.Tensor, torch.Tensor],
        num_heads: int,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        training: bool = True
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Multi-head attention for bicomplex tensors.

    Args:
        query: Query tensor, shape (batch, seq_len, d_model)
        key: Key tensor, shape (batch, seq_len, d_model)
        value: Value tensor, shape (batch, seq_len, d_model)
        num_heads: Number of attention heads
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        training: Whether in training mode

    Returns:
        Tuple of:
            - Output tensor in idempotent form
            - Attention weights averaged across heads
    """
    if not is_idempotent(query):
        raise ValueError("Query must be a bicomplex tensor in idempotent form")
    if not is_idempotent(key):
        raise ValueError("Key must be a bicomplex tensor in idempotent form")
    if not is_idempotent(value):
        raise ValueError("Value must be a bicomplex tensor in idempotent form")

    batch_size = query[0].shape[0]
    seq_len = query[0].shape[1]
    d_model = query[0].shape[2]

    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

    d_k = d_model // num_heads

    # Reshape for multi-head: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
    def reshape_for_heads(tensor):
        e1, e2 = tensor
        e1 = e1.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        e2 = e2.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        return (e1, e2)

    query_heads = reshape_for_heads(query)
    key_heads = reshape_for_heads(key)
    value_heads = reshape_for_heads(value)

    # Flatten batch and heads: (batch, num_heads, seq_len, d_k) -> (batch*num_heads, seq_len, d_k)
    def flatten_heads(tensor):
        e1, e2 = tensor
        e1 = e1.contiguous().view(batch_size * num_heads, seq_len, d_k)
        e2 = e2.contiguous().view(batch_size * num_heads, seq_len, d_k)
        return (e1, e2)

    query_flat = flatten_heads(query_heads)
    key_flat = flatten_heads(key_heads)
    value_flat = flatten_heads(value_heads)

    # Expand attention mask for multiple heads if needed
    if attn_mask is not None and attn_mask.dim() == 2:
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
    elif attn_mask is not None and attn_mask.dim() == 3:
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        attn_mask = attn_mask.contiguous().view(batch_size * num_heads, seq_len, seq_len)

    # Apply scaled dot-product attention
    attn_output, attn_weights = scaled_dot_product_attention(
        query_flat, key_flat, value_flat,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        training=training
    )

    # Reshape back: (batch*num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
    e1, e2 = attn_output
    e1 = e1.view(batch_size, num_heads, seq_len, d_k)
    e2 = e2.view(batch_size, num_heads, seq_len, d_k)
    e1 = e1.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    e2 = e2.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

    output = (e1, e2)

    # Average attention weights across heads
    attn_weights = attn_weights.view(batch_size, num_heads, seq_len, seq_len)
    attn_weights_avg = attn_weights.mean(dim=1)

    return output, attn_weights_avg


def additive_attention(
        query: tuple[torch.Tensor, torch.Tensor],
        key: tuple[torch.Tensor, torch.Tensor],
        value: tuple[torch.Tensor, torch.Tensor],
        v_weight: tuple[torch.Tensor, torch.Tensor],
        dropout_p: float = 0.0,
        training: bool = True
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Additive (Bahdanau) attention for bicomplex tensors.

    Computes: score = v^T * tanh(W_q*q + W_k*k)

    Args:
        query: Query tensor, shape (batch, seq_len_q, d_model)
        key: Key tensor, shape (batch, seq_len_k, d_model)
        value: Value tensor, shape (batch, seq_len_k, d_v)
        v_weight: Learnable weight vector in idempotent form, shape (d_model,)
        dropout_p: Dropout probability
        training: Whether in training mode

    Returns:
        Tuple of:
            - Output tensor in idempotent form
            - Attention weights
    """
    if not is_idempotent(query):
        raise ValueError("Query must be a bicomplex tensor in idempotent form")
    if not is_idempotent(key):
        raise ValueError("Key must be a bicomplex tensor in idempotent form")
    if not is_idempotent(value):
        raise ValueError("Value must be a bicomplex tensor in idempotent form")
    if not is_idempotent(v_weight):
        raise ValueError("v_weight must be a bicomplex tensor in idempotent form")

    batch_size = query[0].shape[0]
    seq_len_q = query[0].shape[1]
    seq_len_k = key[0].shape[1]

    # Expand query and key for broadcasting
    # query: (batch, seq_len_q, 1, d_model)
    # key: (batch, 1, seq_len_k, d_model)
    query_exp = (query[0].unsqueeze(2), query[1].unsqueeze(2))
    key_exp = (key[0].unsqueeze(1), key[1].unsqueeze(1))

    # Add query and key
    combined = add_idempotent(query_exp, key_exp)  # (batch, seq_len_q, seq_len_k, d_model)

    # Apply tanh activation component-wise
    combined_tanh = (torch.tanh(combined[0]), torch.tanh(combined[1]))

    # Compute scores using v_weight
    # Expand v_weight: (d_model,) -> (1, 1, 1, d_model)
    v_exp = (v_weight[0].view(1, 1, 1, -1), v_weight[1].view(1, 1, 1, -1))

    # Element-wise multiply and sum over d_model dimension
    scores_bc = multiply_idempotent(combined_tanh, v_exp)
    scores = modulus(scores_bc)  # Convert to real scores
    scores = scores.sum(dim=-1)  # (batch, seq_len_q, seq_len_k)

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0 and training:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=training)

    # Apply attention to values
    output_e1 = torch.matmul(attn_weights, value[0])
    output_e2 = torch.matmul(attn_weights, value[1])

    output = (output_e1, output_e2)

    return output, attn_weights


def self_attention(
        input: tuple[torch.Tensor, torch.Tensor],
        w_q: tuple[torch.Tensor, torch.Tensor],
        w_k: tuple[torch.Tensor, torch.Tensor],
        w_v: tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        training: bool = True
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Self-attention for bicomplex tensors.

    Computes Q, K, V from the same input, then applies scaled dot-product attention.

    Args:
        input: Input tensor, shape (batch, seq_len, d_model)
        w_q: Query projection weights, shape (d_model, d_k)
        w_k: Key projection weights, shape (d_model, d_k)
        w_v: Value projection weights, shape (d_model, d_v)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        training: Whether in training mode

    Returns:
        Tuple of:
            - Output tensor in idempotent form
            - Attention weights
    """
    if not is_idempotent(input):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(w_q):
        raise ValueError("w_q must be a bicomplex tensor in idempotent form")
    if not is_idempotent(w_k):
        raise ValueError("w_k must be a bicomplex tensor in idempotent form")
    if not is_idempotent(w_v):
        raise ValueError("w_v must be a bicomplex tensor in idempotent form")

    # Project input to query, key, value
    query = matmul(input, w_q)
    key = matmul(input, w_k)
    value = matmul(input, w_v)

    # Apply scaled dot-product attention
    return scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        training=training
    )


def cross_attention(
        query_input: tuple[torch.Tensor, torch.Tensor],
        kv_input: tuple[torch.Tensor, torch.Tensor],
        w_q: tuple[torch.Tensor, torch.Tensor],
        w_k: tuple[torch.Tensor, torch.Tensor],
        w_v: tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        training: bool = True
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Cross-attention for bicomplex tensors.

    Query comes from one input, keys and values from another (e.g., encoder-decoder attention).

    Args:
        query_input: Query input tensor, shape (batch, seq_len_q, d_model_q)
        kv_input: Key/value input tensor, shape (batch, seq_len_kv, d_model_kv)
        w_q: Query projection weights
        w_k: Key projection weights
        w_v: Value projection weights
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        training: Whether in training mode

    Returns:
        Tuple of:
            - Output tensor in idempotent form
            - Attention weights
    """
    if not is_idempotent(query_input):
        raise ValueError("Query input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(kv_input):
        raise ValueError("KV input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(w_q):
        raise ValueError("w_q must be a bicomplex tensor in idempotent form")
    if not is_idempotent(w_k):
        raise ValueError("w_k must be a bicomplex tensor in idempotent form")
    if not is_idempotent(w_v):
        raise ValueError("w_v must be a bicomplex tensor in idempotent form")

    # Project inputs
    query = matmul(query_input, w_q)
    key = matmul(kv_input, w_k)
    value = matmul(kv_input, w_v)

    # Apply scaled dot-product attention
    return scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,  # Cross-attention typically not causal
        training=training
    )


def multiplicative_attention(
        query: tuple[torch.Tensor, torch.Tensor],
        key: tuple[torch.Tensor, torch.Tensor],
        value: tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        training: bool = True
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Multiplicative (Luong) attention for bicomplex tensors.

    Similar to scaled dot-product but without scaling factor.
    Score = Q * K^T

    Args:
        query: Query tensor, shape (batch, seq_len_q, d_model)
        key: Key tensor, shape (batch, seq_len_k, d_model)
        value: Value tensor, shape (batch, seq_len_k, d_v)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        training: Whether in training mode

    Returns:
        Tuple of:
            - Output tensor in idempotent form
            - Attention weights
    """
    if not is_idempotent(query):
        raise ValueError("Query must be a bicomplex tensor in idempotent form")
    if not is_idempotent(key):
        raise ValueError("Key must be a bicomplex tensor in idempotent form")
    if not is_idempotent(value):
        raise ValueError("Value must be a bicomplex tensor in idempotent form")

    # Conjugate and transpose keys
    key_conj = conjugate_idempotent(key)
    key_conj_t = transpose(key_conj, -2, -1)

    # Compute scores
    scores = matmul(query, key_conj_t)
    attn_scores = modulus(scores)

    # Apply mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        else:
            attn_scores = attn_scores + attn_mask

    # Softmax
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # Dropout
    if dropout_p > 0.0 and training:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=training)

    # Apply to values
    output_e1 = torch.matmul(attn_weights, value[0])
    output_e2 = torch.matmul(attn_weights, value[1])

    return (output_e1, output_e2), attn_weights


def local_attention(
        query: tuple[torch.Tensor, torch.Tensor],
        key: tuple[torch.Tensor, torch.Tensor],
        value: tuple[torch.Tensor, torch.Tensor],
        window_size: int,
        dropout_p: float = 0.0,
        training: bool = True
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Local attention with a fixed window size for bicomplex tensors.

    Each query position only attends to keys within a local window.

    Args:
        query: Query tensor, shape (batch, seq_len, d_model)
        key: Key tensor, shape (batch, seq_len, d_model)
        value: Value tensor, shape (batch, seq_len, d_v)
        window_size: Size of the attention window (on each side)
        dropout_p: Dropout probability
        training: Whether in training mode

    Returns:
        Tuple of:
            - Output tensor in idempotent form
            - Attention weights
    """
    if not is_idempotent(query):
        raise ValueError("Query must be a bicomplex tensor in idempotent form")
    if not is_idempotent(key):
        raise ValueError("Key must be a bicomplex tensor in idempotent form")
    if not is_idempotent(value):
        raise ValueError("Value must be a bicomplex tensor in idempotent form")

    seq_len = query[0].shape[1]

    # Create local attention mask
    positions = torch.arange(seq_len, device=query[0].device)
    distances = positions.unsqueeze(0) - positions.unsqueeze(1)
    local_mask = torch.abs(distances) > window_size

    # Apply scaled dot-product attention with local mask
    return scaled_dot_product_attention(
        query, key, value,
        attn_mask=local_mask,
        dropout_p=dropout_p,
        is_causal=False,
        training=training
    )