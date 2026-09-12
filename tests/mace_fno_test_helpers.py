"""Shared, dependency-light fixtures for coupling and configuration tests.

The fake MACE exposes position derivatives and irrep metadata without loading a
real checkpoint. Its hand-written features are not a physical equivariant MACE,
so it must not serve as a rotation-symmetry reference. Keep specialized doubles
next to their tests, and return fresh tensors from every graph factory.
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from mace_fno.cli.config import parse_arguments

DTYPE = torch.float64


def train_arguments(*options: str) -> argparse.Namespace:
    """Parse a minimal training command, optionally overriding its settings."""
    return parse_arguments(
        ["--mace-model", "model.pt", "--train-file", "train.xyz", *options]
    )


class _FakeIrrep:
    def __init__(self, angular_momentum: int, parity: int) -> None:
        self.l = angular_momentum
        self.p = parity


class FakeIrreps:
    def __init__(self, terms: list[tuple[int, int, int]]) -> None:
        self.terms = [
            (multiplicity, _FakeIrrep(angular_momentum, parity))
            for multiplicity, angular_momentum, parity in terms
        ]
        self.dim = sum(
            multiplicity * (2 * irrep.l + 1) for multiplicity, irrep in self.terms
        )

    def __iter__(self):
        return iter(self.terms)

    def slices(self) -> list[slice]:
        result = []
        start = 0
        for multiplicity, irrep in self.terms:
            stop = start + multiplicity * (2 * irrep.l + 1)
            result.append(slice(start, stop))
            start = stop
        return result


class FakeProduct(nn.Module):
    def __init__(self, irreps: FakeIrreps) -> None:
        super().__init__()
        self.linear = nn.Module()
        self.linear.irreps_out = irreps


class FakeMACE(nn.Module):
    """Small position-dependent module matching the coupling-test contract."""

    def __init__(self, with_irreps: bool = False) -> None:
        super().__init__()
        self.local_scale = nn.Parameter(torch.tensor(0.07, dtype=DTYPE))
        if with_irreps:
            self.products = nn.ModuleList(
                [
                    FakeProduct(FakeIrreps([(2, 0, 1), (1, 1, -1)])),
                    FakeProduct(FakeIrreps([(1, 0, -1), (3, 0, 1)])),
                ]
            )

    def forward(self, data, **kwargs):
        compute_force = bool(kwargs.get("compute_force", False))
        positions = data["positions"]
        positions.requires_grad_(True)
        batch = data["batch"]
        num_graphs = int(batch.max().detach().cpu()) + 1
        atom_energy = self.local_scale * positions.square().sum(dim=1)
        energy = atom_energy.new_zeros(num_graphs).index_add(0, batch, atom_energy)
        species = data["node_attrs"][:, 0]
        radius_squared = positions.square().sum(dim=1)
        node_features = torch.stack(
            (
                radius_squared,
                positions[:, 0],
                species + 0.2 * radius_squared,
                positions[:, 1],
            ),
            dim=1,
        )
        forces = -2.0 * self.local_scale * positions if compute_force else None
        return {"energy": energy, "forces": forces, "node_feats": node_features}


def batch_data() -> dict[str, torch.Tensor]:
    """Return independent tensors for two small periodic atom graphs."""
    positions = torch.tensor(
        (
            (1.31, 2.17, 0.2),
            (4.22, 5.41, -0.1),
            (7.15, 1.82, 0.4),
            (2.26, 3.38, -0.3),
            (6.73, 7.11, 0.1),
        ),
        dtype=DTYPE,
    )
    cells = torch.stack(
        (
            torch.diag(torch.tensor((9.0, 10.0, 18.0), dtype=DTYPE)),
            torch.diag(torch.tensor((9.0, 10.0, 18.0), dtype=DTYPE)),
        )
    )
    return {
        "positions": positions,
        "cell": cells,
        "batch": torch.tensor((0, 0, 0, 1, 1), dtype=torch.long),
        "ptr": torch.tensor((0, 3, 5), dtype=torch.long),
        "node_attrs": torch.tensor(
            ((1.0,), (0.0,), (1.0,), (0.0,), (1.0,)), dtype=DTYPE
        ),
    }
