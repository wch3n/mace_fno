from __future__ import annotations

import unittest

import torch
from torch import nn

from mace_fno import (
    FrozenMACEFeatures,
    MACEFNOResidual,
    NeutralLatentHead,
    energy_force_loss,
    mace_invariant_indices,
)
from examples.train_mace_residual import ensure_frozen_residual_targets

DTYPE = torch.float64


class _FakeIrrep:
    def __init__(self, angular_momentum: int, parity: int) -> None:
        self.l = angular_momentum
        self.p = parity


class _FakeIrreps:
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


class _FakeProduct(nn.Module):
    def __init__(self, irreps: _FakeIrreps) -> None:
        super().__init__()
        self.linear = nn.Module()
        self.linear.irreps_out = irreps


class _AtomicEnergyLookup(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "atomic_energies",
            torch.tensor(((-18599.4376, -688.8681),), dtype=DTYPE),
        )

    def forward(self, node_attrs: torch.Tensor) -> torch.Tensor:
        return node_attrs @ self.atomic_energies.T.to(node_attrs)


class _FakeAtomicMACE(nn.Module):
    """Expose separated interaction energy like a ScaleShiftMACE checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.atomic_energies_fn = _AtomicEnergyLookup()
        self.interaction_scale = nn.Parameter(torch.tensor(0.013, dtype=DTYPE))

    def forward(self, data, **kwargs):
        del kwargs
        positions = data["positions"]
        positions.requires_grad_(True)
        batch = data["batch"]
        num_graphs = int(batch.max().detach().cpu()) + 1
        atomic = self.atomic_energies_fn(data["node_attrs"]).squeeze(1)
        e0 = atomic.new_zeros(num_graphs).index_add(0, batch, atomic)
        atom_interaction = self.interaction_scale * positions.square().sum(dim=1)
        interaction = atom_interaction.new_zeros(num_graphs).index_add(
            0, batch, atom_interaction
        )
        node_features = torch.stack(
            (positions.square().sum(dim=1), positions[:, 0]), dim=1
        )
        return {
            "energy": e0 + interaction,
            "interaction_energy": interaction,
            "node_feats": node_features,
        }


class _FakeMACE(nn.Module):
    """Small position-dependent module matching the relevant MACE contract."""

    def __init__(self, with_irreps: bool = False) -> None:
        super().__init__()
        self.local_scale = nn.Parameter(torch.tensor(0.07, dtype=DTYPE))
        if with_irreps:
            self.products = nn.ModuleList(
                [
                    _FakeProduct(_FakeIrreps([(2, 0, 1), (1, 1, -1)])),
                    _FakeProduct(_FakeIrreps([(1, 0, -1), (3, 0, 1)])),
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


def _batch_data() -> dict[str, torch.Tensor]:
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


class CouplingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)

    def test_irreps_metadata_locates_even_scalars_per_layer(self) -> None:
        indices, descriptor_dim = mace_invariant_indices(_FakeMACE(with_irreps=True))
        self.assertEqual(indices, [0, 1, 6, 7, 8])
        self.assertEqual(descriptor_dim, 9)

    def test_frozen_features_retain_position_derivatives(self) -> None:
        backbone = _FakeMACE()
        adapter = FrozenMACEFeatures(backbone, invariant_indices=(0, 2))
        data = _batch_data()
        _, features, _ = adapter(data)
        derivative = torch.autograd.grad(features.sum(), data["positions"])[0]
        self.assertTrue(torch.isfinite(derivative).all())
        self.assertGreater(derivative.abs().sum().item(), 0.0)
        self.assertFalse(backbone.local_scale.requires_grad)
        adapter.train()
        self.assertFalse(backbone.training)

    def test_float32_base_energy_uses_float64_atomic_reference(self) -> None:
        backbone = _FakeAtomicMACE()
        exact_atomic_energies = (
            backbone.atomic_energies_fn.atomic_energies.detach().clone()
        )
        adapter = FrozenMACEFeatures(backbone, invariant_indices=(0, 1)).float()
        data = {
            "positions": torch.tensor(
                ((0.2, 0.1, 0.0), (1.1, -0.2, 0.3), (0.4, 0.7, -0.1)),
                dtype=torch.float32,
            ),
            "node_attrs": torch.tensor(
                ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
                dtype=torch.float32,
            ),
            "batch": torch.zeros(3, dtype=torch.long),
        }
        corrected, _, output = adapter(data)
        exact_e0 = (data["node_attrs"].double() @ exact_atomic_energies.T).sum()
        expected = exact_e0 + output["interaction_energy"].double().sum()

        torch.testing.assert_close(corrected[0], expected, atol=0.0, rtol=0.0)
        self.assertGreater(
            abs(output["energy"].double()[0].item() - expected.item()),
            1.0e-4,
        )
        self.assertEqual(adapter.atomic_energies_reference.dtype, torch.float64)

    def test_source_head_is_neutral_for_every_graph(self) -> None:
        features = torch.randn((7, 5), dtype=DTYPE)
        batch = torch.tensor((0, 0, 0, 1, 1, 1, 1), dtype=torch.long)
        head = NeutralLatentHead(5, 3, hidden_channels=8).to(dtype=DTYPE)
        sources = head(features, batch)
        sums = sources.new_zeros((2, 3)).index_add(0, batch, sources)
        torch.testing.assert_close(sums, torch.zeros_like(sums), atol=2e-15, rtol=0)

    def test_batched_coupling_and_joint_force_backward(self) -> None:
        data = _batch_data()
        model = MACEFNOResidual(
            _FakeMACE(),
            (12, 12),
            channels=2,
            n_modes=(3, 3),
            source_hidden_channels=8,
            fno_architecture="linear",
            invariant_indices=(0, 2),
            reference_cell=data["cell"][0],
        ).to(dtype=DTYPE)
        model.train()
        output = model(data, training=True, compute_force=True, return_fields=True)
        self.assertEqual(output["energy"].shape, (2,))
        self.assertEqual(output["forces"].shape, data["positions"].shape)
        self.assertEqual(len(output["density"]), 2)
        self.assertEqual(output["density"][0].shape, (2, 12, 12))

        source_sums = (
            output["sources"]
            .new_zeros((2, 2))
            .index_add(0, data["batch"], output["sources"])
        )
        torch.testing.assert_close(
            source_sums, torch.zeros_like(source_sums), atol=2e-15, rtol=0
        )

        target_energy = torch.zeros_like(output["energy"])
        target_forces = torch.zeros_like(output["forces"])
        terms = energy_force_loss(
            output["energy"],
            output["forces"],
            target_energy,
            target_forces,
            data["batch"],
        )
        terms["loss"].backward()
        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(trainable_gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in trainable_gradients))
        self.assertIsNone(model.backbone.mace_model.local_scale.grad)

    def test_batched_hybrid_2p5d_coupling(self) -> None:
        data = _batch_data()
        model = MACEFNOResidual(
            _FakeMACE(),
            (12, 12),
            channels=2,
            n_modes=(3, 3),
            source_hidden_channels=8,
            fno_hidden_channels=4,
            fno_layers=2,
            z_grid_size=6,
            z_extent=6.0,
            fno_z_mixing="global",
            invariant_indices=(0, 2),
            reference_cell=data["cell"][0],
        ).to(dtype=DTYPE)
        model.train()
        output = model(
            data,
            training=True,
            compute_force=False,
            compute_residual_force=True,
            return_fields=True,
        )
        self.assertEqual(model.spatial_scheme, "2.5d")
        self.assertEqual(model.long_range.field_operator.z_mixing, "global")
        self.assertEqual(output["energy"].shape, (2,))
        self.assertEqual(output["residual_forces"].shape, data["positions"].shape)
        self.assertEqual(output["density"][0].shape, (2, 6, 12, 12))
        self.assertTrue(torch.isfinite(output["residual_forces"]).all())

        output["residual_forces"].square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertIsNone(model.backbone.mace_model.local_scale.grad)

    def test_batched_interlaced_2p5d_coupling(self) -> None:
        data = _batch_data()
        model = MACEFNOResidual(
            _FakeMACE(),
            (12, 12),
            channels=2,
            n_modes=(3, 3),
            source_hidden_channels=8,
            fno_hidden_channels=4,
            fno_layers=2,
            z_grid_size=6,
            z_extent=6.0,
            fno_z_mixing="global",
            fno_lateral_interlacing=2,
            fno_planar_symmetry="d4",
            invariant_indices=(0, 2),
            reference_cell=data["cell"][0],
        ).to(dtype=DTYPE)
        output = model(
            data,
            training=True,
            compute_force=False,
            compute_residual_force=True,
        )
        self.assertEqual(output["energy"].shape, (2,))
        self.assertEqual(output["residual_forces"].shape, data["positions"].shape)
        self.assertTrue(torch.isfinite(output["residual_forces"]).all())
        output["residual_forces"].square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_batched_fully_periodic_3d_coupling(self) -> None:
        data = _batch_data()
        model = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=2,
            n_modes=(2, 2),
            source_hidden_channels=8,
            fno_architecture="linear",
            spatial_scheme="3d",
            z_grid_size=6,
            fno_z_modes=2,
            invariant_indices=(0, 2),
            reference_cell=data["cell"][0],
        ).to(dtype=DTYPE)
        output = model(
            data,
            compute_force=False,
            compute_residual_force=True,
            return_fields=True,
        )
        self.assertEqual(model.spatial_scheme, "3d")
        self.assertEqual(output["energy"].shape, (2,))
        self.assertEqual(output["residual_forces"].shape, data["positions"].shape)
        self.assertEqual(output["density"][0].shape, (2, 6, 8, 8))
        self.assertTrue(torch.isfinite(output["residual_forces"]).all())

    def test_batched_interlaced_fully_periodic_3d_coupling(self) -> None:
        data = _batch_data()
        model = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=2,
            n_modes=(2, 2),
            source_hidden_channels=8,
            fno_architecture="linear",
            spatial_scheme="3d",
            z_grid_size=6,
            fno_z_modes=2,
            fno_volume_interlacing=2,
            invariant_indices=(0, 2),
            reference_cell=data["cell"][0],
        ).to(dtype=DTYPE)
        output = model(
            data,
            compute_force=False,
            compute_residual_force=True,
        )
        self.assertEqual(model.long_range.volume_interlacing, 2)
        self.assertEqual(output["energy"].shape, (2,))
        self.assertEqual(output["residual_forces"].shape, data["positions"].shape)
        self.assertTrue(torch.isfinite(output["residual_forces"]).all())

    def test_3d_reference_cell_guard_includes_third_vector(self) -> None:
        data = _batch_data()
        model = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=1,
            n_modes=(2, 2),
            fno_architecture="linear",
            spatial_scheme="3d",
            z_grid_size=6,
            invariant_indices=(0, 2),
            reference_cell=data["cell"][0],
        ).to(dtype=DTYPE)
        data["cell"][1, 2, 2] += 0.1
        with self.assertRaisesRegex(ValueError, "fixed-cell"):
            model(data, compute_force=False)

    def test_eqgino_3d_coupling_requires_cubic_geometry(self) -> None:
        cubic_cell = 8.0 * torch.eye(3, dtype=DTYPE)
        model = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=2,
            n_modes=(2, 2),
            source_hidden_channels=8,
            fno_hidden_channels=4,
            fno_layers=1,
            spatial_scheme="3d",
            z_grid_size=8,
            fno_z_modes=2,
            fno_spectral_symmetry="eqgino",
            fno_spectral_groups=2,
            invariant_indices=(0, 2),
            reference_cell=cubic_cell,
        ).to(dtype=DTYPE)
        self.assertEqual(
            model.long_range.field_operator.spectral_symmetry, "eqgino"
        )
        self.assertEqual(model.long_range.field_operator.spectral_groups, 2)

        with self.assertRaisesRegex(ValueError, "cubic reference_cell"):
            MACEFNOResidual(
                _FakeMACE(),
                (8, 8),
                channels=1,
                n_modes=(2, 2),
                spatial_scheme="3d",
                z_grid_size=8,
                fno_spectral_symmetry="eqgino",
                invariant_indices=(0, 2),
                reference_cell=torch.diag(
                    torch.tensor((8.0, 8.0, 9.0), dtype=DTYPE)
                ),
            )

        unreferenced = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=1,
            n_modes=(2, 2),
            spatial_scheme="3d",
            z_grid_size=8,
            fno_spectral_symmetry="eqgino",
            invariant_indices=(0, 2),
        ).to(dtype=DTYPE)
        with self.assertRaisesRegex(ValueError, "cubic cells"):
            unreferenced(_batch_data(), compute_force=False)

    def test_reported_force_is_energy_gradient(self) -> None:
        data = _batch_data()
        one_graph = {
            key: value[:3].clone() if key in {"positions", "node_attrs"} else value
            for key, value in data.items()
        }
        one_graph["batch"] = torch.zeros(3, dtype=torch.long)
        one_graph["ptr"] = torch.tensor((0, 3), dtype=torch.long)
        one_graph["cell"] = data["cell"][:1].clone()
        model = MACEFNOResidual(
            _FakeMACE(),
            (12, 12),
            channels=1,
            n_modes=(3, 3),
            source_hidden_channels=8,
            fno_architecture="linear",
            invariant_indices=(0, 2),
        ).to(dtype=DTYPE)
        model.eval()
        output = model(one_graph, training=False, compute_force=True)

        step = 1.0e-6
        plus = {key: value.clone() for key, value in one_graph.items()}
        minus = {key: value.clone() for key, value in one_graph.items()}
        plus["positions"][0, 1] += step
        minus["positions"][0, 1] -= step
        plus_energy = model(plus, compute_force=False)["energy"]
        minus_energy = model(minus, compute_force=False)["energy"]
        finite_difference = -(plus_energy - minus_energy) / (2.0 * step)
        torch.testing.assert_close(
            output["forces"][0, 1],
            finite_difference[0],
            atol=2e-8,
            rtol=2e-6,
        )

    def test_base_and_residual_forces_add_to_total_force(self) -> None:
        data = _batch_data()
        model = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=1,
            n_modes=(2, 2),
            fno_architecture="linear",
            invariant_indices=(0, 2),
        ).to(dtype=DTYPE)
        model.eval()
        output = model(
            data,
            training=False,
            compute_force=True,
            compute_residual_force=True,
            compute_base_force=True,
        )
        torch.testing.assert_close(
            output["forces"],
            output["base_forces"] + output["residual_forces"],
            atol=2e-12,
            rtol=2e-12,
        )

    def test_frozen_targets_are_cached_as_reference_minus_mace(self) -> None:
        data = _batch_data()
        graphs = []
        for graph_index, (start, stop) in enumerate(((0, 3), (3, 5))):
            graphs.append(
                {
                    "positions": data["positions"][start:stop].clone(),
                    "cell": data["cell"][graph_index : graph_index + 1].clone(),
                    "batch": torch.zeros(stop - start, dtype=torch.long),
                    "ptr": torch.tensor((0, stop - start), dtype=torch.long),
                    "node_attrs": data["node_attrs"][start:stop].clone(),
                }
            )
        backbone = _FakeMACE()
        samples = []
        for graph in graphs:
            base_energy = (
                backbone.local_scale.detach() * graph["positions"].square().sum()
            ).reshape(1)
            base_forces = -2.0 * backbone.local_scale.detach() * graph["positions"]
            samples.append(
                {
                    "data": graph,
                    "energy": base_energy + 0.25,
                    "forces": base_forces + 0.125,
                    "num_atoms": graph["positions"].shape[0],
                    "formula": "X",
                }
            )
        model = MACEFNOResidual(
            backbone,
            (8, 8),
            channels=1,
            n_modes=(2, 2),
            fno_architecture="linear",
            invariant_indices=(0, 2),
        ).to(dtype=DTYPE)

        self.assertTrue(
            ensure_frozen_residual_targets(
                model, samples, device=torch.device("cpu"), batch_size=2
            )
        )
        for sample in samples:
            torch.testing.assert_close(
                sample["residual_energy"],
                torch.full((1,), 0.25, dtype=DTYPE),
            )
            torch.testing.assert_close(
                sample["residual_forces"],
                torch.full_like(sample["forces"], 0.125),
            )
        self.assertFalse(
            ensure_frozen_residual_targets(
                model, samples, device=torch.device("cpu"), batch_size=2
            )
        )

    def test_reference_cell_guard_rejects_changed_cell(self) -> None:
        data = _batch_data()
        model = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=1,
            n_modes=(2, 2),
            fno_architecture="linear",
            invariant_indices=(0, 2),
            reference_cell=data["cell"][0],
        ).to(dtype=DTYPE)
        data["cell"][1, 0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "fixed-cell"):
            model(data, compute_force=False)


if __name__ == "__main__":
    unittest.main()
