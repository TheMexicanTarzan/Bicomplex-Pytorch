"""
Tensorial operations for bicomplex numbers in idempotent form.

All operations work on bicomplex tensors represented as tuples (e1, e2)
where e1 and e2 are complex PyTorch tensors.
"""

import torch
from typing import Optional, Union

from representations import is_idempotent


def matmul(a: tuple[torch.Tensor, torch.Tensor],
           b: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Matrix multiplication for bicomplex tensors in idempotent form.

    Args:
        a: Bicomplex tensor, each component shape (..., M, N)
        b: Bicomplex tensor, each component shape (..., N, P)

    Returns:
        Tuple of tensors with shape (..., M, P)
    """
    if not is_idempotent(a):
        raise ValueError("First argument must be a bicomplex tensor in idempotent form")
    if not is_idempotent(b):
        raise ValueError("Second argument must be a bicomplex tensor in idempotent form")

    return (torch.matmul(a[0], b[0]), torch.matmul(a[1], b[1]))


def bmm(a: tuple[torch.Tensor, torch.Tensor],
        b: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch matrix multiplication for bicomplex tensors.

    Args:
        a: Bicomplex tensor, each component shape (B, M, N)
        b: Bicomplex tensor, each component shape (B, N, P)

    Returns:
        Tuple of tensors with shape (B, M, P)
    """
    if not is_idempotent(a):
        raise ValueError("First argument must be a bicomplex tensor in idempotent form")
    if not is_idempotent(b):
        raise ValueError("Second argument must be a bicomplex tensor in idempotent form")

    return (torch.bmm(a[0], b[0]), torch.bmm(a[1], b[1]))


def transpose(tensor: tuple[torch.Tensor, torch.Tensor],
              dim0: int, dim1: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose two dimensions of a bicomplex tensor."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (tensor[0].transpose(dim0, dim1), tensor[1].transpose(dim0, dim1))


def sum(tensor: tuple[torch.Tensor, torch.Tensor],
        dim: Optional[Union[int, tuple]] = None,
        keepdim: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum bicomplex tensor along specified dimensions."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.sum(tensor[0], dim=dim, keepdim=keepdim),
            torch.sum(tensor[1], dim=dim, keepdim=keepdim))


def mean(tensor: tuple[torch.Tensor, torch.Tensor],
         dim: Optional[Union[int, tuple]] = None,
         keepdim: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of bicomplex tensor along dimensions."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.mean(tensor[0], dim=dim, keepdim=keepdim),
            torch.mean(tensor[1], dim=dim, keepdim=keepdim))


def norm(tensor: tuple[torch.Tensor, torch.Tensor],
         p: Union[str, float] = 'fro',
         dim: Optional[Union[int, tuple]] = None,
         keepdim: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute norm of bicomplex tensor component-wise.

    Returns norm of each idempotent component separately.
    """
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.norm(tensor[0], p=p, dim=dim, keepdim=keepdim),
            torch.norm(tensor[1], p=p, dim=dim, keepdim=keepdim))


def reshape(tensor: tuple[torch.Tensor, torch.Tensor],
            *shape) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape bicomplex tensor."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (tensor[0].reshape(shape), tensor[1].reshape(shape))


def flatten(tensor: tuple[torch.Tensor, torch.Tensor],
            start_dim: int = 0,
            end_dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten bicomplex tensor dimensions."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.flatten(tensor[0], start_dim=start_dim, end_dim=end_dim),
            torch.flatten(tensor[1], start_dim=start_dim, end_dim=end_dim))


def cat(tensors: list[tuple[torch.Tensor, torch.Tensor]],
        dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate bicomplex tensors along a dimension."""
    for i, t in enumerate(tensors):
        if not is_idempotent(t):
            raise ValueError(f"Tensor {i} must be a bicomplex tensor in idempotent form")

    e1_list = [t[0] for t in tensors]
    e2_list = [t[1] for t in tensors]

    return (torch.cat(e1_list, dim=dim), torch.cat(e2_list, dim=dim))


def stack(tensors: list[tuple[torch.Tensor, torch.Tensor]],
          dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack bicomplex tensors along a new dimension."""
    for i, t in enumerate(tensors):
        if not is_idempotent(t):
            raise ValueError(f"Tensor {i} must be a bicomplex tensor in idempotent form")

    e1_list = [t[0] for t in tensors]
    e2_list = [t[1] for t in tensors]

    return (torch.stack(e1_list, dim=dim), torch.stack(e2_list, dim=dim))


def split(tensor: tuple[torch.Tensor, torch.Tensor],
          split_size_or_sections: Union[int, list],
          dim: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split bicomplex tensor into chunks."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1_splits = torch.split(tensor[0], split_size_or_sections, dim=dim)
    e2_splits = torch.split(tensor[1], split_size_or_sections, dim=dim)

    return list(zip(e1_splits, e2_splits))


def einsum(equation: str,
           *operands: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Einstein summation for bicomplex tensors.

    Applies einsum independently to e1 and e2 components.
    """
    for i, op in enumerate(operands):
        if not is_idempotent(op):
            raise ValueError(f"Operand {i} must be a bicomplex tensor in idempotent form")

    e1_operands = [op[0] for op in operands]
    e2_operands = [op[1] for op in operands]

    result_e1 = torch.einsum(equation, *e1_operands)
    result_e2 = torch.einsum(equation, *e2_operands)

    return (result_e1, result_e2)


def permute(tensor: tuple[torch.Tensor, torch.Tensor],
            *dims) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute dimensions of bicomplex tensor."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (tensor[0].permute(*dims), tensor[1].permute(*dims))


def squeeze(tensor: tuple[torch.Tensor, torch.Tensor],
            dim: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove dimensions of size 1."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.squeeze(tensor[0], dim=dim), torch.squeeze(tensor[1], dim=dim))


def unsqueeze(tensor: tuple[torch.Tensor, torch.Tensor],
              dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Add dimension of size 1."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.unsqueeze(tensor[0], dim=dim), torch.unsqueeze(tensor[1], dim=dim))


def view(tensor: tuple[torch.Tensor, torch.Tensor],
         *shape) -> tuple[torch.Tensor, torch.Tensor]:
    """Return view of bicomplex tensor with new shape."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (tensor[0].view(shape), tensor[1].view(shape))


def clone(tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a copy of bicomplex tensor."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (tensor[0].clone(), tensor[1].clone())


def detach(tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Detach bicomplex tensor from computation graph."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (tensor[0].detach(), tensor[1].detach())


def contiguous(tensor: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous bicomplex tensor in memory."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (tensor[0].contiguous(), tensor[1].contiguous())


def expand(tensor: tuple[torch.Tensor, torch.Tensor],
           *sizes) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand tensor to larger size."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (tensor[0].expand(*sizes), tensor[1].expand(*sizes))


def repeat(tensor: tuple[torch.Tensor, torch.Tensor],
           *sizes) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat tensor along specified dimensions."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (tensor[0].repeat(*sizes), tensor[1].repeat(*sizes))


def chunk(tensor: tuple[torch.Tensor, torch.Tensor],
          chunks: int,
          dim: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split tensor into specific number of chunks."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    e1_chunks = torch.chunk(tensor[0], chunks, dim=dim)
    e2_chunks = torch.chunk(tensor[1], chunks, dim=dim)

    return list(zip(e1_chunks, e2_chunks))


def gather(tensor: tuple[torch.Tensor, torch.Tensor],
           dim: int,
           index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather values along an axis specified by dim."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.gather(tensor[0], dim, index),
            torch.gather(tensor[1], dim, index))


def scatter(tensor: tuple[torch.Tensor, torch.Tensor],
            dim: int,
            index: torch.Tensor,
            src: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter values along an axis specified by dim."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(src):
        raise ValueError("Source must be a bicomplex tensor in idempotent form")

    return (torch.scatter(tensor[0], dim, index, src[0]),
            torch.scatter(tensor[1], dim, index, src[1]))


def masked_select(tensor: tuple[torch.Tensor, torch.Tensor],
                  mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select elements according to boolean mask."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.masked_select(tensor[0], mask),
            torch.masked_select(tensor[1], mask))


def masked_fill(tensor: tuple[torch.Tensor, torch.Tensor],
                mask: torch.Tensor,
                value: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill elements of tensor with value where mask is True."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")
    if not is_idempotent(value):
        raise ValueError("Value must be a bicomplex tensor in idempotent form")

    return (torch.masked_fill(tensor[0], mask, value[0]),
            torch.masked_fill(tensor[1], mask, value[1]))


def where(condition: torch.Tensor,
          x: tuple[torch.Tensor, torch.Tensor],
          y: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Select elements from x or y depending on condition."""
    if not is_idempotent(x):
        raise ValueError("First tensor must be a bicomplex tensor in idempotent form")
    if not is_idempotent(y):
        raise ValueError("Second tensor must be a bicomplex tensor in idempotent form")

    return (torch.where(condition, x[0], y[0]),
            torch.where(condition, x[1], y[1]))


def index_select(tensor: tuple[torch.Tensor, torch.Tensor],
                 dim: int,
                 index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select elements along dim using indices."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.index_select(tensor[0], dim, index),
            torch.index_select(tensor[1], dim, index))


def take(tensor: tuple[torch.Tensor, torch.Tensor],
         index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Take elements from tensor at flat indices."""
    if not is_idempotent(tensor):
        raise ValueError("Input must be a bicomplex tensor in idempotent form")

    return (torch.take(tensor[0], index), torch.take(tensor[1], index))