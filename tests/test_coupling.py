from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from mace_fno import (
    FrozenMACEFeatures,
    MACEFNOResidual,
    NeutralLatentHead,
    ParticleMeshLongRange,
    energy_force_loss,
    mace_invariant_indices,
)
from mace_fno.training import (
    amplitude_convergence_diagnostic,
    build_mace_fno_model,
    checkpoint_model_parameters,
    ensure_frozen_residual_targets,
    infer_checkpoint_z_mixing,
    low_k_response_diagnostic,
    mace_state_dict,
    residual_state_dict,
    resolve_checkpoint_model_path,
)

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

    def test_joint_mode_backpropagates_total_force_loss_into_mace(self) -> None:
        data = _batch_data()
        backbone = _FakeMACE()
        model = MACEFNOResidual(
            backbone,
            (8, 8),
            channels=1,
            n_modes=(2, 2),
            fno_architecture="linear",
            invariant_indices=(0, 2),
            reference_cell=data["cell"][0],
            mace_training="joint",
        ).to(dtype=DTYPE)
        model.train()
        output = model(data, training=True, compute_force=True)
        loss = output["energy"].square().mean() + output["forces"].square().mean()
        loss.backward()

        self.assertTrue(backbone.local_scale.requires_grad)
        self.assertTrue(backbone.training)
        self.assertIsNotNone(backbone.local_scale.grad)
        self.assertTrue(torch.isfinite(backbone.local_scale.grad))

        model.eval()
        self.assertFalse(backbone.training)

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

    def test_low_k_diagnostic_is_validation_only_for_periodic_3d(self) -> None:
        data = _batch_data()
        cubic_cell = 8.0 * torch.eye(3, dtype=DTYPE)
        graph = {
            "positions": data["positions"][:3].clone(),
            "cell": cubic_cell.unsqueeze(0),
            "batch": torch.zeros(3, dtype=torch.long),
            "ptr": torch.tensor((0, 3), dtype=torch.long),
            "node_attrs": data["node_attrs"][:3].clone(),
        }
        sample = {
            "data": graph,
            "energy": torch.zeros(1, dtype=DTYPE),
            "forces": torch.zeros((3, 3), dtype=DTYPE),
            "num_atoms": 3,
            "formula": "X",
        }
        model = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=1,
            n_modes=(2, 2),
            fno_architecture="linear",
            spatial_scheme="3d",
            z_grid_size=8,
            fno_z_modes=2,
            invariant_indices=(0, 2),
            reference_cell=cubic_cell,
        ).to(dtype=DTYPE)
        model.train()

        report = low_k_response_diagnostic(
            model,
            [sample],
            max_mode=1,
            fit_shells=3,
            field_batch_size=8,
        )
        self.assertTrue(model.training)
        self.assertEqual(report["spatial_scheme"], "3d")
        self.assertEqual(report["diagnostic_kind"], "periodic_3d")
        self.assertEqual(report["samples"], 1)
        self.assertEqual(len(report["per_sample_response"][0]["modes"]), 13)

        planar_model = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=1,
            n_modes=(2, 2),
            fno_architecture="linear",
            invariant_indices=(0, 2),
            reference_cell=cubic_cell,
        ).to(dtype=DTYPE)
        planar_report = low_k_response_diagnostic(
            planar_model, [sample], fit_shells=2
        )
        self.assertEqual(planar_report["diagnostic_kind"], "planar_2d")
        self.assertEqual(planar_report["spatial_scheme"], "2d")
        self.assertEqual(len(planar_report["per_sample_response"][0]["modes"]), 4)
        self.assertNotIn(
            "mean_low_k_coulomb_template_relative_error", planar_report
        )

        planar_model.long_range = ParticleMeshLongRange(
            (8, 8), deconvolve_assignment=False
        ).to(dtype=DTYPE)
        analytic_planar_report = low_k_response_diagnostic(
            planar_model, [sample], max_mode=2, fit_shells=4
        )
        analytic_fit = analytic_planar_report["low_k_planar_response_fit"]
        self.assertIsNotNone(analytic_fit)
        assert analytic_fit is not None
        self.assertAlmostEqual(
            analytic_fit["free_power_exponent_p"], 1.0, places=10
        )
        self.assertAlmostEqual(
            analytic_fit["reference_power_log_r2"], 1.0, places=10
        )
        amplitude_report = amplitude_convergence_diagnostic(
            planar_model,
            [sample],
            relative_amplitudes=(0.025, 0.05, 0.1),
            max_mode=2,
            fit_shells=4,
        )
        amplitude_summary = amplitude_report["summary"]
        self.assertEqual(amplitude_report["estimated_field_evaluations"], 108)
        self.assertTrue(amplitude_summary["curvature_stable_within_tolerance"])
        self.assertLess(amplitude_summary["maximum_mode_relative_span"], 1.0e-8)
        for exponent in amplitude_summary["free_power_exponents"]:
            self.assertAlmostEqual(exponent, 1.0, places=10)

        with self.assertRaisesRegex(ValueError, "distinct"):
            amplitude_convergence_diagnostic(
                planar_model,
                [sample],
                relative_amplitudes=(0.05, 0.05),
            )

        slab_model = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=1,
            n_modes=(2, 2),
            fno_architecture="linear",
            z_grid_size=6,
            z_extent=6.0,
            invariant_indices=(0, 2),
            reference_cell=cubic_cell,
        ).to(dtype=DTYPE)
        slab_report = low_k_response_diagnostic(
            slab_model, [sample], fit_shells=2, z_profiles=3
        )
        self.assertEqual(slab_report["diagnostic_kind"], "slab_2p5d")
        self.assertEqual(
            slab_report["z_profile_names"], ["monopole", "dipole", "quadrupole"]
        )
        self.assertEqual(len(slab_report["per_sample_response"][0]["modes"]), 4)
        self.assertEqual(slab_report["probes_per_mode"], 3)

        monopole_report = low_k_response_diagnostic(
            slab_model, [sample], fit_shells=2, z_profiles=1
        )
        self.assertIsNone(
            monopole_report["mean_low_k_coulomb_template_relative_error"]
        )

        anisotropic_model = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=1,
            n_modes=(2, 2),
            fno_architecture="linear",
            spatial_scheme="3d",
            z_grid_size=8,
            fno_z_modes=2,
            invariant_indices=(0, 2),
            reference_cell=data["cell"][0],
        ).to(dtype=DTYPE)
        anisotropic_graph = dict(graph)
        anisotropic_graph["cell"] = data["cell"][:1].clone()
        anisotropic_sample = dict(sample)
        anisotropic_sample["data"] = anisotropic_graph
        anisotropic_report = low_k_response_diagnostic(
            anisotropic_model, [anisotropic_sample]
        )
        self.assertEqual(anisotropic_report["diagnostic_kind"], "periodic_3d")
        self.assertGreater(
            anisotropic_report["per_sample_response"][0]["physical_shells"], 3
        )

        interlaced_model = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=1,
            n_modes=(2, 2),
            fno_architecture="linear",
            spatial_scheme="3d",
            z_grid_size=8,
            fno_z_modes=2,
            fno_volume_interlacing=2,
            invariant_indices=(0, 2),
            reference_cell=cubic_cell,
        ).to(dtype=DTYPE)
        interlaced_report = low_k_response_diagnostic(
            interlaced_model, [sample]
        )
        self.assertEqual(interlaced_report["diagnostic_kind"], "periodic_3d")

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

    def test_isotropic_3d_cell_mode_accepts_uniform_scalings(self) -> None:
        data = _batch_data()
        data["cell"] = torch.stack(
            (8.0 * torch.eye(3, dtype=DTYPE), 10.0 * torch.eye(3, dtype=DTYPE))
        )
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
            invariant_indices=(0, 2),
            reference_cell=data["cell"][0],
            cell_mode="isotropic",
        ).to(dtype=DTYPE)
        output = model(data, compute_force=False, compute_residual_force=True)
        self.assertEqual(output["energy"].shape, (2,))
        self.assertTrue(torch.isfinite(output["residual_forces"]).all())
        self.assertEqual(
            model.long_range.field_operator.cell_conditioning, "isotropic"
        )

        invalid = {key: value.clone() for key, value in data.items()}
        invalid["cell"][1, 2, 2] = 11.0
        with self.assertRaisesRegex(ValueError, "uniform scaling"):
            model(invalid, compute_force=False)

        with self.assertRaisesRegex(ValueError, "requires a nonlinear"):
            MACEFNOResidual(
                _FakeMACE(),
                (8, 8),
                channels=1,
                n_modes=(2, 2),
                fno_architecture="linear",
                spatial_scheme="3d",
                z_grid_size=8,
                invariant_indices=(0, 2),
                cell_mode="isotropic",
            )

    def test_anisotropic_3d_cell_mode_accepts_variable_cell_shapes(self) -> None:
        data = _batch_data()
        data["cell"] = torch.stack(
            (
                torch.diag(torch.tensor((8.0, 9.0, 10.0), dtype=DTYPE)),
                torch.tensor(
                    ((8.4, 0.0, 0.0), (0.3, 9.2, 0.0), (0.1, -0.2, 10.7)),
                    dtype=DTYPE,
                ),
            )
        )
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
            invariant_indices=(0, 2),
            reference_cell=data["cell"][0],
            cell_mode="anisotropic",
        ).to(dtype=DTYPE)
        output = model(data, compute_force=False, compute_residual_force=True)
        self.assertEqual(output["energy"].shape, (2,))
        self.assertTrue(torch.isfinite(output["residual_forces"]).all())
        self.assertEqual(
            model.long_range.field_operator.cell_conditioning, "anisotropic"
        )

        invalid = {key: value.clone() for key, value in data.items()}
        invalid["cell"][1, 2] = invalid["cell"][1, 1]
        with self.assertRaisesRegex(ValueError, "nonsingular"):
            model(invalid, compute_force=False)

        metric = MACEFNOResidual(
            _FakeMACE(),
            (8, 8),
            channels=2,
            n_modes=(2, 2),
            fno_hidden_channels=4,
            spatial_scheme="3d",
            z_grid_size=8,
            fno_z_modes=2,
            fno_spectral_symmetry="metric_eqgino",
            fno_spectral_groups=2,
            fno_metric_hidden_channels=5,
            invariant_indices=(0, 2),
            cell_mode="anisotropic",
        ).to(dtype=DTYPE)
        metric_output = metric(
            data, compute_force=False, compute_residual_force=True
        )
        self.assertTrue(torch.isfinite(metric_output["residual_forces"]).all())
        self.assertEqual(
            metric.long_range.field_operator.metric_hidden_channels, 5
        )

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

    def test_legacy_checkpoint_reconstructs_2p5d_model(self) -> None:
        reference_cell = _batch_data()["cell"][0]
        original = MACEFNOResidual(
            _FakeMACE(with_irreps=True),
            (8, 10),
            channels=2,
            n_modes=(2, 3),
            source_hidden_channels=7,
            fno_hidden_channels=5,
            fno_layers=2,
            spatial_scheme="2.5d",
            z_grid_size=6,
            z_extent=7.5,
            fno_z_mixing="global",
            reference_cell=reference_cell,
        ).to(dtype=DTYPE)
        checkpoint = {
            "residual_state_dict": residual_state_dict(original),
            "grid_shape": (8, 10),
            "n_modes": (2, 3),
            "spatial_scheme": "2.5d",
            "z_grid_size": 6,
            "z_extent": 7.5,
            "z_center": "mean",
            "z_kernel_size": 3,
            "channels": 2,
            "source_hidden_channels": 7,
            "fno_hidden_channels": 5,
            "fno_layers": 2,
            "architecture": "nonlinear",
            "reference_cell": reference_cell,
            "dtype": "float64",
        }

        self.assertEqual(infer_checkpoint_z_mixing(checkpoint), "global")
        parameters = checkpoint_model_parameters(checkpoint)
        self.assertEqual(parameters["cell_mode"], "fixed")
        self.assertEqual(parameters["fno_lateral_interlacing"], 1)
        self.assertEqual(parameters["fno_planar_symmetry"], "none")

        restored = build_mace_fno_model(
            checkpoint,
            _FakeMACE(with_irreps=True),
        )
        self.assertFalse(restored.training)
        self.assertEqual(restored.spatial_scheme, "2.5d")
        self.assertEqual(restored.long_range.field_operator.z_mixing, "global")
        for key, expected in checkpoint["residual_state_dict"].items():
            torch.testing.assert_close(residual_state_dict(restored)[key], expected)

    def test_checkpoint_parameters_recover_3d_layout(self) -> None:
        checkpoint = {
            "residual_state_dict": {},
            "grid_shape": (12, 14),
            "n_modes": (3, 4, 5),
            "spatial_scheme": "3d",
            "cell_mode": "isotropic",
            "z_grid_size": 10,
            "volume_interlacing": 1,
            "spectral_symmetry": "metric_eqgino",
            "spectral_groups": 4,
            "metric_hidden_channels": 12,
            "channels": 2,
            "source_hidden_channels": 8,
            "fno_hidden_channels": 6,
            "fno_layers": 2,
            "architecture": "nonlinear",
            "reference_cell": torch.eye(3, dtype=DTYPE),
        }
        parameters = checkpoint_model_parameters(checkpoint)
        self.assertEqual(parameters["n_modes"], (4, 5))
        self.assertEqual(parameters["fno_z_modes"], 3)
        self.assertEqual(parameters["fno_spectral_symmetry"], "metric_eqgino")
        self.assertEqual(parameters["fno_spectral_groups"], 4)
        self.assertEqual(parameters["fno_metric_hidden_channels"], 12)
        self.assertEqual(parameters["fno_interlacing_training"], "full")

        checkpoint["cell_mode"] = "anisotropic"
        checkpoint["spectral_symmetry"] = "none"
        checkpoint["spectral_groups"] = 1
        parameters = checkpoint_model_parameters(checkpoint)
        self.assertEqual(parameters["cell_mode"], "anisotropic")

        checkpoint.pop("metric_hidden_channels")
        parameters = checkpoint_model_parameters(checkpoint)
        self.assertEqual(parameters["fno_metric_hidden_channels"], 16)

    def test_metric_eqgino_checkpoint_reconstructs_model(self) -> None:
        reference_cell = _batch_data()["cell"][0]
        original = MACEFNOResidual(
            _FakeMACE(with_irreps=True),
            (8, 8),
            channels=2,
            n_modes=(2, 2),
            source_hidden_channels=7,
            fno_hidden_channels=4,
            fno_layers=1,
            spatial_scheme="3d",
            z_grid_size=8,
            fno_z_modes=2,
            fno_spectral_symmetry="metric_eqgino",
            fno_spectral_groups=2,
            fno_metric_hidden_channels=5,
            reference_cell=reference_cell,
            cell_mode="anisotropic",
        ).to(dtype=DTYPE)
        checkpoint = {
            "residual_state_dict": residual_state_dict(original),
            "grid_shape": (8, 8),
            "n_modes": (2, 2, 2),
            "spatial_scheme": "3d",
            "cell_mode": "anisotropic",
            "z_grid_size": 8,
            "volume_interlacing": 1,
            "spectral_symmetry": "metric_eqgino",
            "spectral_groups": 2,
            "metric_hidden_channels": 5,
            "channels": 2,
            "source_hidden_channels": 7,
            "fno_hidden_channels": 4,
            "fno_layers": 1,
            "architecture": "nonlinear",
            "reference_cell": reference_cell,
            "dtype": "float64",
        }
        restored = build_mace_fno_model(
            checkpoint, _FakeMACE(with_irreps=True), dtype=DTYPE
        )
        self.assertEqual(
            restored.long_range.field_operator.spectral_symmetry,
            "metric_eqgino",
        )
        self.assertEqual(
            restored.long_range.field_operator.metric_hidden_channels,
            5,
        )
        for key, expected in checkpoint["residual_state_dict"].items():
            torch.testing.assert_close(residual_state_dict(restored)[key], expected)

    def test_joint_checkpoint_restores_updated_mace_state(self) -> None:
        reference_cell = _batch_data()["cell"][0]
        original = MACEFNOResidual(
            _FakeMACE(with_irreps=True),
            (8, 8),
            channels=1,
            n_modes=(2, 2),
            fno_architecture="linear",
            reference_cell=reference_cell,
            mace_training="joint",
        ).to(dtype=DTYPE)
        original.backbone.mace_model.local_scale.data.fill_(0.321)
        checkpoint = {
            "checkpoint_format_version": 2,
            "mace_training": "joint",
            "mace_state_dict": mace_state_dict(original),
            "residual_state_dict": residual_state_dict(original),
            "grid_shape": (8, 8),
            "n_modes": (2, 2),
            "spatial_scheme": "2d",
            "channels": 1,
            "source_hidden_channels": 64,
            "fno_hidden_channels": 32,
            "fno_layers": 4,
            "architecture": "linear",
            "reference_cell": reference_cell,
            "dtype": "float64",
        }

        restored = build_mace_fno_model(
            checkpoint, _FakeMACE(with_irreps=True), dtype=DTYPE
        )

        self.assertEqual(restored.mace_training, "joint")
        self.assertAlmostEqual(
            restored.backbone.mace_model.local_scale.item(), 0.321
        )

    def test_legacy_artifact_model_path_follows_relocated_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs"
            checkpoint = run_root / "les_au_mgo" / "residual.pt"
            model = (
                run_root
                / "les_au_mgo"
                / "pretrained"
                / "Au2-MgO_stagetwo.model"
            )
            checkpoint.parent.mkdir(parents=True)
            model.parent.mkdir(parents=True)
            checkpoint.touch()
            model.touch()
            stored = (
                Path("/old/repository/artifacts")
                / "les_au_mgo"
                / "pretrained"
                / model.name
            )
            self.assertEqual(
                resolve_checkpoint_model_path(stored, checkpoint), model
            )


if __name__ == "__main__":
    unittest.main()
