"""Linear physical-cell operators: superposition, symmetry, and force training."""

import unittest

import torch
from mace_fno_test_helpers import FakeMACE, batch_data, train_arguments

from mace_fno import MACEFNOResidual, SlabFNOFieldOperator2D
from mace_fno.training.configuration import TrainingConfig
from mace_fno.training.initialization import initialize_zero_residual


class LinearMetricTests(unittest.TestCase):
    def test_linear_slab_superposition_d4_and_height_coupling(self):
        torch.manual_seed(41)
        op = SlabFNOFieldOperator2D(
            2,
            3,
            (2, 2),
            architecture="linear",
            spectral_symmetry="metric_eqgino",
            metric_reference_length=8,
        ).double()
        cell = torch.diag(torch.tensor([8.0, 8.0, 20.0], dtype=torch.float64))
        x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
        y = torch.randn_like(x)
        torch.testing.assert_close(
            op(2 * x - 3 * y, cell), 2 * op(x, cell) - 3 * op(y, cell)
        )
        torch.testing.assert_close(op(torch.zeros_like(x), cell), torch.zeros_like(x))
        for flip in (False, True):
            for rotation in range(4):

                def transform(v):
                    return torch.rot90(v.flip(-1) if flip else v, rotation, (-2, -1))

                torch.testing.assert_close(
                    op(transform(x), cell), transform(op(x, cell))
                )
        # Set only a first-height -> last-height matrix entry. The compound
        # index is channel-major, with the height index varying fastest.
        with torch.no_grad():
            op.fno.spectral.radial_weight.zero_()
            op.fno.spectral.radial_weight[0, 0, 2, :] = 1
        x.zero_()
        x[0, 0] = 1
        expected = torch.zeros_like(x)
        expected[0, 2] = 1
        torch.testing.assert_close(op(x, cell), expected)

    def test_variable_cells_force_gradients_zero_initialization_and_reload(self):
        for scheme, cell_mode in (
            ("3d", "isotropic"),
            ("3d", "anisotropic"),
            ("2.5d", "anisotropic"),
        ):
            for training in ("frozen", "joint"):
                with self.subTest(
                    scheme=scheme, cell_mode=cell_mode, training=training
                ):
                    graph = batch_data()
                    if cell_mode == "isotropic":
                        graph["cell"] = torch.stack(
                            [torch.eye(3) * 9, torch.eye(3) * 10]
                        ).double()
                    kwargs = dict(
                        channels=2,
                        n_modes=(2, 2),
                        source_hidden_channels=4,
                        fno_architecture="linear",
                        fno_spectral_symmetry="metric_eqgino",
                        spatial_scheme=scheme,
                        cell_mode=cell_mode,
                        z_grid_size=8,
                        invariant_indices=(0, 2),
                        reference_cell=graph["cell"][0],
                        mace_training=training,
                    )
                    if scheme == "2.5d":
                        kwargs["z_extent"] = 8.0
                    model = MACEFNOResidual(FakeMACE(), (8, 8), **kwargs).double()
                    if scheme == "3d":
                        self.assertEqual(
                            model.long_range.field_operator.cell_conditioning, "none"
                        )
                    # Forward/force backward must include the source head and
                    # the MACE backbone in joint training.
                    result = model(graph, compute_force=True, training=True)
                    (
                        result["energy"].square().sum()
                        + result["forces"].square().sum()
                    ).backward()
                    grads = [p.grad for p in model.parameters() if p.requires_grad]
                    self.assertTrue(
                        grads
                        and all(
                            g is not None and torch.isfinite(g).all() for g in grads
                        )
                    )
                    restored = MACEFNOResidual(FakeMACE(), (8, 8), **kwargs).double()
                    restored.load_state_dict(model.state_dict())
                    torch.testing.assert_close(
                        restored(graph, compute_force=True)["forces"], result["forces"]
                    )
                    # Energy-force consistency on a representative coordinate.
                    delta = 1.0e-5
                    energies = []
                    for sign in (-1, 1):
                        shifted = {k: v.detach().clone() for k, v in graph.items()}
                        shifted["positions"][0, 0] += sign * delta
                        energies.append(
                            model(shifted, compute_force=False)["energy"].sum()
                        )
                    torch.testing.assert_close(
                        -(energies[1] - energies[0]) / (2 * delta),
                        result["forces"][0, 0],
                        atol=1.0e-6,
                        rtol=1.0e-5,
                    )
                    initialize_zero_residual(model)
                    zero = model(graph, compute_force=True, training=True)
                    torch.testing.assert_close(zero["energy"], zero["base_energy"])
                    model.zero_grad()
                    zero["energy"].sum().backward()
                    self.assertTrue(
                        any(
                            p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.long_range.parameters()
                        )
                    )

    def test_training_configuration_accepts_variable_linear_metric_cells(self):
        for scheme, mode in (
            ("3d", "isotropic"),
            ("3d", "anisotropic"),
            ("2.5d", "anisotropic"),
        ):
            args = [
                "--spatial-scheme",
                scheme,
                "--cell-mode",
                mode,
                "--architecture",
                "linear",
                "--spectral-symmetry",
                "metric_eqgino",
                "--z-grid",
                "24",
            ]
            if scheme == "2.5d":
                args += ["--z-extent", "22"]
            cfg = TrainingConfig.from_namespace(train_arguments(*args))
            self.assertEqual(cfg.model.architecture, "linear")


if __name__ == "__main__":
    unittest.main()
