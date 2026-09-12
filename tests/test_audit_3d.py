from __future__ import annotations

import argparse
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch import nn

from mace_fno import LearnedParticleMeshLongRange3D
from mace_fno.cli import audit_3d

DTYPE = torch.float64
CPU = torch.device("cpu")


class CubicDiscretizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transforms = audit_3d.cubic_transformations(DTYPE, CPU)

    def statuses(self, grid=(32, 24, 24), modes=(6, 6, 6), symmetry="metric_eqgino"):
        return {
            name: audit_3d.cubic_discretization_status(transform, grid, modes, symmetry)
            for name, transform in self.transforms.items()
        }

    def test_exact_symmetry_requires_compatible_mesh_modes_and_operator(self) -> None:
        all_transforms = {"c4_x", "c4_y", "c4_z", "cycle_xyz", "swap_xy", "inversion"}
        normal_subgroup = {"c4_z", "swap_xy", "inversion"}
        x_subgroup = {"c4_x", "inversion"}
        cases = (
            (
                "equal axes",
                (24,) * 3,
                (6,) * 3,
                "metric_eqgino",
                all_transforms,
                all_transforms,
            ),
            (
                "normal mesh refinement",
                (32, 24, 24),
                (6,) * 3,
                "metric_eqgino",
                normal_subgroup,
                all_transforms,
            ),
            (
                "normal mode refinement",
                (24,) * 3,
                (9, 6, 6),
                "metric_eqgino",
                all_transforms,
                normal_subgroup,
            ),
            (
                "internal zxy axis order",
                (24, 32, 24),
                (6, 9, 6),
                "metric_eqgino",
                x_subgroup,
                x_subgroup,
            ),
            (
                "all unequal axes",
                (32, 24, 28),
                (6, 4, 5),
                "metric_eqgino",
                {"inversion"},
                {"inversion"},
            ),
            (
                "unconstrained operator",
                (24,) * 3,
                (6,) * 3,
                "none",
                all_transforms,
                all_transforms,
            ),
        )
        self.assertEqual(set(self.transforms), all_transforms)
        for case, grid, modes, symmetry, mesh_symmetries, mode_symmetries in cases:
            for name, status in self.statuses(grid, modes, symmetry).items():
                with self.subTest(case=case, transformation=name):
                    preserves_mesh = name in mesh_symmetries
                    preserves_modes = name in mode_symmetries
                    expected_exact = (
                        preserves_mesh
                        and preserves_modes
                        and symmetry == "metric_eqgino"
                    )
                    self.assertEqual(status["preserves_mesh"], preserves_mesh)
                    self.assertEqual(status["preserves_mode_cutoff"], preserves_modes)
                    self.assertEqual(status["expected_exact"], expected_exact)
                    reasons = status["diagnostic_only_reasons"]
                    self.assertEqual(not reasons, expected_exact)
                    if not preserves_mesh:
                        self.assertTrue(
                            any("unequal mesh sizes" in reason for reason in reasons)
                        )
                    if not preserves_modes:
                        self.assertTrue(
                            any(
                                "unequal Fourier cutoffs" in reason
                                for reason in reasons
                            )
                        )

    def test_invalid_transform_is_not_silently_classified(self) -> None:
        with self.assertRaisesRegex(ValueError, "signed permutation"):
            audit_3d.cubic_discretization_status(
                torch.ones(3, 3), (8,) * 3, (2,) * 3, "metric_eqgino"
            )

    def test_strict_aggregation_excludes_only_unsupported_transformations(self) -> None:
        report = self.statuses()
        for item in report.values():
            error = 1e-9 if item["expected_exact"] else 100.0
            item.update(
                residual_energy_change_mev=error,
                residual_force_equivariance_rmse_mev_per_angstrom=error,
            )
        checks = audit_3d.spatial_exact_checks(report, None, DTYPE)
        for check in checks.values():
            self.assertLess(check["observed"], check["threshold"])
            self.assertEqual(
                set(check["transformations"]), {"c4_z", "swap_xy", "inversion"}
            )
        # A genuine failure in the compatible subgroup must remain a failure.
        report["c4_z"]["residual_energy_change_mev"] = 1.0
        checks = audit_3d.spatial_exact_checks(report, None, DTYPE)
        check = checks["cubic_residual_energy_invariance"]
        self.assertGreater(check["observed"], check["threshold"])

    def test_rigid_rotation_is_strict_without_any_cubic_geometry(self) -> None:
        report = {
            "oblique_rotation": {
                "expected_exact": True,
                "residual_energy_change_mev": 1.0,
                "residual_force_equivariance_rmse_mev_per_angstrom": 1.0,
            }
        }
        checks = audit_3d.spatial_exact_checks(None, report, DTYPE)
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(c["observed"] > c["threshold"] for c in checks.values()))


class SymmetryGeometryTests(unittest.TestCase):
    def test_selection_finds_actual_cube_without_cubic_metadata(self) -> None:
        rotation = audit_3d.rigid_cell_rotation(DTYPE, CPU)
        samples = [
            {
                "benchmark_group": "cubic",
                "data": {
                    "cell": torch.diag(torch.tensor([8.0, 9.0, 10.0], dtype=DTYPE))
                },
            },
            {"benchmark_group": "Cl32Li32", "data": {"cell": 8.0 * rotation}},
        ]
        for seed in range(5):
            self.assertEqual(audit_3d.select_sample_indices(samples, 1, seed), [1])
            indices = audit_3d.select_sample_indices(samples, 8, seed)
            self.assertEqual(indices[0], 1)
            self.assertEqual(sorted(indices), [0, 1])

    def test_empty_cache_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            audit_3d.select_sample_indices([], 1, 17)

    def test_whole_cell_rotation_preserves_fractional_geometry_and_image_indices(
        self,
    ) -> None:
        cell = torch.tensor(
            [[8.0, 0.2, 0.0], [0.5, 9.0, 0.1], [0.0, 0.3, 10.0]], dtype=DTYPE
        )
        fractional = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], dtype=DTYPE)
        unit = torch.tensor([[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE)
        graph = {
            "positions": fractional @ cell,
            "cell": cell.clone(),
            "shifts": unit @ cell,
            "unit_shifts": unit.clone(),
        }
        rotation = audit_3d.rigid_cell_rotation(DTYPE, CPU)
        torch.testing.assert_close(rotation @ rotation.T, torch.eye(3, dtype=DTYPE))
        self.assertAlmostEqual(float(torch.linalg.det(rotation)), 1.0)
        audit_3d.transform_graph(graph, rotation, rotate_cell=True)
        torch.testing.assert_close(
            graph["positions"] @ torch.linalg.inv(graph["cell"]), fractional
        )
        torch.testing.assert_close(graph["unit_shifts"], unit, atol=0, rtol=0)
        torch.testing.assert_close(
            graph["shifts"], graph["unit_shifts"] @ graph["cell"]
        )

    def test_fixed_cell_rotation_updates_neighbor_image_indices(self) -> None:
        cell = 8.0 * torch.eye(3, dtype=DTYPE)
        unit = torch.tensor([[1.0, -1.0, 0.0]], dtype=DTYPE)
        graph = {
            "positions": torch.tensor([[1.0, 2.0, 3.0]], dtype=DTYPE),
            "cell": cell.clone(),
            "shifts": unit @ cell,
            "unit_shifts": unit.clone(),
        }
        rotation = audit_3d.cubic_transformations(DTYPE, CPU)["c4_y"]
        audit_3d.transform_graph(graph, rotation, rotate_cell=False)
        torch.testing.assert_close(graph["cell"], cell, atol=0, rtol=0)
        torch.testing.assert_close(graph["shifts"], graph["unit_shifts"] @ cell)
        torch.testing.assert_close(graph["unit_shifts"], unit @ rotation.T)


class _ParticleMeshResidual(nn.Module):
    """Exercise the complete checker without requiring an installed MACE model."""

    spatial_scheme = "3d"
    cell_mode = "anisotropic"

    def __init__(
        self,
        *,
        break_rotation=False,
        grid=(12, 8, 8),
        modes=(2, 2, 2),
        interlacing=1,
        cell_mode="anisotropic",
    ):
        super().__init__()
        self.break_rotation = break_rotation
        self.cell_mode = cell_mode
        self.long_range = LearnedParticleMeshLongRange3D(
            grid,
            channels=1,
            n_modes=modes,
            hidden_channels=3,
            n_layers=1,
            spectral_symmetry="metric_eqgino",
            metric_reference_length=8.0,
            volume_interlacing=interlacing,
        ).to(dtype=DTYPE)

    def forward(
        self,
        graph,
        *,
        training=False,
        compute_force=False,
        compute_residual_force=False,
        compute_base_force=False,
    ):
        positions = graph["positions"].requires_grad_(True)
        cells = graph["cell"].reshape(-1, 3, 3)
        sources = graph["node_attrs"]
        energy = self.long_range(positions, sources, cells, batch=graph["batch"])
        if self.break_rotation:
            off_diagonal = cells - torch.diag_embed(cells.diagonal(dim1=-2, dim2=-1))
            energy = energy + 0.01 * (off_diagonal.abs().sum((-2, -1)) > 1e-5)
        forces = None
        if compute_force or compute_residual_force:
            forces = -torch.autograd.grad(energy.sum(), positions)[0]
        return {
            "energy": energy,
            "residual_energy": energy,
            "base_energy": torch.zeros_like(energy),
            "sources": sources,
            "forces": forces,
            "residual_forces": forces,
            "base_forces": torch.zeros_like(positions),
        }


class Audit3DIntegrationTests(unittest.TestCase):
    def run_audit(
        self,
        *,
        break_rotation=False,
        triclinic=False,
        grid=(12, 8, 8),
        modes=(2, 2, 2),
        interlacing=1,
        cell_mode="anisotropic",
    ):
        torch.manual_seed(91)
        model = _ParticleMeshResidual(
            break_rotation=break_rotation,
            grid=grid,
            modes=modes,
            interlacing=interlacing,
            cell_mode=cell_mode,
        )
        cell = 8.0 * torch.eye(3, dtype=DTYPE)
        if triclinic:
            cell[1] = torch.tensor((0.5, 9.0, 0.2), dtype=DTYPE)
        positions = torch.tensor(
            [[1.13, 2.27, 3.41], [5.17, 6.31, 4.53], [2.39, 5.67, 6.81]], dtype=DTYPE
        )
        graph = {
            "positions": positions,
            "cell": cell,
            "shifts": torch.empty(0, 3, dtype=DTYPE),
            "node_attrs": torch.tensor([[1.0], [-0.4], [-0.6]], dtype=DTYPE),
            "batch": torch.zeros(3, dtype=torch.long),
            "ptr": torch.tensor([0, 3]),
        }
        sample = {
            "data": graph,
            "num_atoms": 3,
            "energy": torch.zeros(1, dtype=DTYPE),
            "forces": torch.zeros_like(positions),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache.pt"
            torch.save({"samples": [sample]}, cache)
            checkpoint = {
                "test_cache": str(cache),
                "grid_shape": grid[1:],
                "z_grid_size": grid[0],
                "n_modes": modes,
                "spectral_symmetry": "metric_eqgino",
                "metric_parameterization": "shell_spline",
                "dtype": "float64",
            }
            args = argparse.Namespace(
                checkpoint=root / "model.pt",
                sample_cache=None,
                samples=1,
                translation=0.1,
                fd_step=1e-4,
                fd_components=3,
                seed=17,
                device="cpu",
                strict=True,
                output=root / "report.json",
            )
            with (
                patch.object(audit_3d, "parse_arguments", return_value=args),
                patch.object(
                    audit_3d, "load_mace_fno_model", return_value=(model, checkpoint)
                ),
                redirect_stdout(io.StringIO()),
            ):
                if break_rotation:
                    with self.assertRaisesRegex(
                        RuntimeError, "rigid_cell_rotation_residual_energy_invariance"
                    ):
                        audit_3d.main()
                else:
                    audit_3d.main()
            return json.loads(args.output.read_text())

    def test_strict_pass_keeps_anisotropic_mesh_errors_visible(self) -> None:
        report = self.run_audit()
        self.assertTrue(
            all(c["passed"] for c in report["promised_exact_checks"].values())
        )
        cubic = report["cubic_signed_axis_transformations"]
        self.assertEqual(set(cubic), set(audit_3d.cubic_transformations(DTYPE, CPU)))
        self.assertFalse(cubic["c4_x"]["expected_exact"])
        self.assertGreater(cubic["c4_x"]["residual_energy_change_mev"], 1e-5)
        self.assertTrue(cubic["c4_z"]["expected_exact"])
        self.assertIsNotNone(report["rigid_cell_rotation"])

    def test_triclinic_cells_do_not_skip_rigid_rotation(self) -> None:
        report = self.run_audit(triclinic=True)
        self.assertIsNone(report["cubic_signed_axis_transformations"])
        self.assertTrue(
            report["promised_exact_checks"][
                "rigid_cell_rotation_residual_force_covariance"
            ]["passed"]
        )

    def test_equal_mesh_and_modes_still_enforce_all_cubic_transformations(self) -> None:
        report = self.run_audit(grid=(8, 8, 8))
        self.assertEqual(
            report["diagnostic_status"]["cubic_diagnostic_only_transformations"], []
        )
        check = report["promised_exact_checks"]["cubic_residual_energy_invariance"]
        self.assertEqual(len(check["transformations"]), 6)
        self.assertTrue(check["passed"])

    def test_unequal_modes_on_cubic_mesh_are_not_full_cubic_symmetry(self) -> None:
        report = self.run_audit(grid=(8, 8, 8), modes=(3, 2, 2))
        item = report["cubic_signed_axis_transformations"]["c4_x"]
        self.assertTrue(item["preserves_mesh"])
        self.assertFalse(item["preserves_mode_cutoff"])
        self.assertFalse(item["expected_exact"])
        self.assertGreater(item["residual_energy_change_mev"], 1e-5)

    def test_interlacing_preserves_compatible_subgroup(self) -> None:
        report = self.run_audit(interlacing=2)
        self.assertTrue(
            all(check["passed"] for check in report["promised_exact_checks"].values())
        )

    def test_reference_cell_guards_are_not_bypassed_for_rotation_check(self) -> None:
        for cell_mode in ("fixed", "isotropic"):
            with self.subTest(cell_mode=cell_mode):
                report = self.run_audit(grid=(8, 8, 8), cell_mode=cell_mode)
                self.assertIsNone(report["rigid_cell_rotation"])
                self.assertIn(
                    "Not evaluated", report["symmetry_scope"]["rigid_cell_rotation"]
                )
                self.assertTrue(
                    report["promised_exact_checks"]["cubic_residual_force_covariance"][
                        "passed"
                    ]
                )

    def test_broken_rigid_rotation_still_fails_strict_mode(self) -> None:
        report = self.run_audit(break_rotation=True)
        self.assertFalse(
            report["promised_exact_checks"][
                "rigid_cell_rotation_residual_energy_invariance"
            ]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
