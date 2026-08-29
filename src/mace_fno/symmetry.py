"""Small symmetry utilities for fully periodic cubic particle meshes."""

from __future__ import annotations

from itertools import permutations, product

import torch
from torch import Tensor


def cubic_signed_permutation_matrices(
    *,
    include_reflections: bool = True,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the 24-element ``O`` or 48-element ``O_h`` cubic point group.

    Each matrix is a signed permutation acting on components expressed in the
    orthonormal cell-axis basis. With ``include_reflections=False``, only
    determinant +1 operations are retained.
    """
    resolved_dtype = dtype or torch.get_default_dtype()
    matrices = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = torch.zeros((3, 3), dtype=resolved_dtype, device=device)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if include_reflections or bool(torch.linalg.det(matrix) > 0):
                matrices.append(matrix)
    return torch.stack(matrices)


def is_cubic_cell(cell: Tensor, *, tolerance: float = 1.0e-6) -> bool:
    """Return whether three cell vectors are orthogonal and equally long."""
    if cell.shape != (3, 3) or not torch.is_floating_point(cell):
        raise ValueError("cell must be a floating-point tensor with shape (3, 3)")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    lengths = torch.linalg.vector_norm(cell, dim=1)
    if bool((lengths <= torch.finfo(cell.dtype).eps).any().detach().cpu()):
        return False
    gram = cell @ cell.T
    off_diagonal = gram - torch.diag(torch.diagonal(gram))
    scale = lengths.max().clamp_min(1.0)
    return bool(
        (
            (lengths.max() - lengths.min() <= tolerance * scale)
            & (off_diagonal.abs().max() <= tolerance * scale.square())
        )
        .detach()
        .cpu()
    )


def transform_in_cell_axis_basis(
    vectors: Tensor,
    cell: Tensor,
    transformations: Tensor,
) -> Tensor:
    """Apply one or more orthogonal cell-axis transformations to vectors.

    Args:
        vectors: Cartesian row vectors with shape ``(..., 3)``.
        cell: Three Cartesian cell row vectors with shape ``(3, 3)``.
        transformations: One matrix ``(3, 3)`` or a batch ``(group, 3, 3)``
            acting on components in the normalized cell-axis basis.

    Returns:
        ``vectors.shape`` for one transformation, or
        ``(group, *vectors.shape)`` for a batch.
    """
    if vectors.shape[-1:] != (3,):
        raise ValueError("vectors must have final dimension 3")
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    if transformations.shape[-2:] != (3, 3) or transformations.ndim not in {2, 3}:
        raise ValueError("transformations must have shape (3, 3) or (group, 3, 3)")
    if not all(torch.is_floating_point(value) for value in (vectors, cell, transformations)):
        raise TypeError("vectors, cell and transformations must be floating point")
    if not (vectors.device == cell.device == transformations.device):
        raise ValueError("vectors, cell and transformations must share a device")
    if not (vectors.dtype == cell.dtype == transformations.dtype):
        raise ValueError("vectors, cell and transformations must share a dtype")

    basis = cell / torch.linalg.vector_norm(cell, dim=1, keepdim=True)
    if transformations.ndim == 2:
        cartesian = basis.T @ transformations @ basis
        return torch.einsum("ij,...j->...i", cartesian, vectors)
    cartesian = torch.einsum(
        "ia,gab,bj->gij", basis.T, transformations, basis
    )
    return torch.einsum("gij,...j->g...i", cartesian, vectors)
