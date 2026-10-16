"""Full R(|k_parallel|, z, z') layers and matched linear/nonlinear stacks."""

import tempfile
import unittest
from itertools import product
from pathlib import Path

import torch
import yaml
from ase import Atoms
from mace_fno_test_helpers import FakeIrreps, FakeMACE, FakeProduct, batch_data
from torch import nn

from mace_fno import MACEFNOCalculator, SlabFNOFieldOperator2D
from mace_fno.cli.config import parse_arguments
from mace_fno.fno_slab import SlabFNOBlock2D
from mace_fno.training import build_mace_fno_model
from mace_fno.training.checkpoint import (
    infer_checkpoint_z_mixing,
    training_checkpoint_payload,
)
from mace_fno.training.configuration import TrainingConfig
from mace_fno.training.initialization import (
    configure_output_projection_warmup,
    finish_output_projection_warmup,
    initialize_scaled_residual_output,
    initialize_zero_residual,
)
from mace_fno.training.setup import PreparedData, build_training_model
from mace_fno.training.trainer import OptimizationResult

DTYPE = torch.float64


class FullHeightFieldTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(742)
        self.cell = torch.diag(torch.tensor([8.0, 8.0, 20.0], dtype=DTYPE))

    def operator(self, architecture, **kwargs):
        return SlabFNOFieldOperator2D(
            2,
            3,
            (2, 2),
            hidden_channels=4,
            n_layers=2,
            z_mixing="full",
            architecture=architecture,
            metric_reference_length=8,
            **kwargs,
        ).double()

    def test_full_spectral_matrix_has_independent_height_and_wavevector_entries(self):
        block = SlabFNOBlock2D(
            2,
            3,
            (2, 2),
            z_mixing="full",
            activation="identity",
            spectral_symmetry="metric_eqgino",
            metric_reference_length=8,
        ).double()
        self.assertIsNone(block.z_mixing)
        with torch.no_grad():
            block.local.weight.zero_()
            weights = block.spectral.spectral.radial_weight
            weights.zero_()
            # Compound indices are (channel * nz + height). Shell knots: 0,1,2.
            weights[0, 0, 5, 1] = 2.0  # channel 0,z=0 -> channel 1,z=2 at |n|=1
            weights[0, 0, 1, 2] = -3.0  # -> channel 0,z=1 at |n|=sqrt(2)
        phase = 2 * torch.pi * torch.arange(8, dtype=DTYPE) / 8
        first = phase[:, None].cos().expand(8, 8)
        second = (phase[:, None] + phase[None, :]).cos()
        field = torch.zeros(1, 2, 3, 8, 8, dtype=DTYPE)
        field[0, 0, 0] = first + second
        expected = torch.zeros_like(field)
        expected[0, 1, 2] = 2 * first
        expected[0, 0, 1] = -3 * second
        torch.testing.assert_close(
            block(field, self.cell), expected, atol=2e-14, rtol=2e-14
        )
        # No circular-height symmetry: moving the input off z=0 removes its response.
        torch.testing.assert_close(
            block(field.roll(1, dims=2), self.cell), torch.zeros_like(field)
        )

    def test_same_parameters_but_only_identity_stack_is_linear(self):
        for symmetry, parameterization in (
            ("none", "shell_spline"),
            ("metric_eqgino", "shell_spline"),
            ("metric_eqgino", "radial_mlp"),
        ):
            with self.subTest(symmetry=symmetry, parameterization=parameterization):
                kwargs = dict(
                    spectral_symmetry=symmetry, metric_parameterization=parameterization
                )
                nonlinear = self.operator("nonlinear", **kwargs)
                linear = self.operator("linear", **kwargs)
                linear.load_state_dict(nonlinear.state_dict(), strict=True)
                self.assertEqual(
                    sum(p.numel() for p in linear.parameters()),
                    sum(p.numel() for p in nonlinear.parameters()),
                )
                x = torch.randn(2, 3, 8, 8, dtype=DTYPE)
                y = torch.randn_like(x)
                torch.testing.assert_close(
                    linear(2 * x - 3 * y, self.cell),
                    2 * linear(x, self.cell) - 3 * linear(y, self.cell),
                    atol=1e-13,
                    rtol=1e-12,
                )
                for op in (linear, nonlinear):
                    torch.testing.assert_close(
                        op(torch.zeros_like(x), self.cell), torch.zeros_like(x)
                    )
                self.assertGreater(
                    (nonlinear(2 * x, self.cell) - 2 * nonlinear(x, self.cell))
                    .abs()
                    .max()
                    .item(),
                    1e-8,
                )

    def test_metric_symmetry_batch_cells_and_gradients(self):
        for architecture in ("linear", "nonlinear"):
            with self.subTest(architecture=architecture):
                op = self.operator(
                    architecture, spectral_symmetry="metric_eqgino", spectral_groups=2
                )
                x = torch.randn(2, 3, 8, 8, dtype=DTYPE)
                value = op(x, self.cell)
                for flip, rotation in product((False, True), range(4)):

                    def transform(v):
                        return torch.rot90(
                            v.flip(-1) if flip else v, rotation, (-2, -1)
                        )

                    torch.testing.assert_close(
                        op(transform(x), self.cell), transform(value)
                    )
                torch.testing.assert_close(
                    op(x.roll((2, 3), (-2, -1)), self.cell),
                    value.roll((2, 3), (-2, -1)),
                )
                skew = self.cell.clone()
                skew[1, :2] = skew.new_tensor([2.0, 10.0])
                cells = torch.stack((self.cell, skew)).requires_grad_()
                fields = torch.stack((x, 2 * x)).requires_grad_()
                result = op(fields, cells)
                torch.testing.assert_close(
                    result, torch.stack([op(f, c) for f, c in zip(fields, cells)])
                )
                other_vacuum = cells.detach().clone()
                other_vacuum[:, 2] *= 2
                torch.testing.assert_close(result, op(fields, other_vacuum))
                gradients = torch.autograd.grad(
                    result.square().sum(), (fields, cells), create_graph=True
                )
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                sum(g.square().sum() for g in gradients).backward()
                self.assertTrue(
                    all(
                        p.grad is not None and torch.isfinite(p.grad).all()
                        for p in op.parameters()
                    )
                )


class FullHeightIntegrationTests(unittest.TestCase):
    @staticmethod
    def backbone():
        mace = FakeMACE()
        mace.products = nn.ModuleList([FakeProduct(FakeIrreps([(4, 0, 1)]))])
        return mace

    def configuration(
        self, path, architecture="nonlinear", training="frozen", **overrides
    ):
        document = {
            "mace_model": "local.model",
            "train_file": "train.xyz",
            "spatial_scheme": "2.5d",
            "cell_mode": "anisotropic",
            "grid": 8,
            "modes": 2,
            "z_grid": 6,
            "z_extent": 10.0,
            "z_mixing": "full",
            "channels": 2,
            "source_hidden_channels": 4,
            "fno_hidden_channels": 4,
            "fno_layers": 2,
            "spectral_symmetry": "metric_eqgino",
            "spectral_groups": 2,
            "lateral_interlacing": 2,
            "architecture": architecture,
            "mace_training": training,
        }
        document.update(overrides)
        path.write_text(yaml.safe_dump(document))
        return TrainingConfig.from_namespace(parse_arguments(["--config", str(path)]))

    @staticmethod
    def graph():
        graph = batch_data()
        graph["cell"][1, 1, :2] = graph["cell"].new_tensor([2.0, 12.0])
        return graph

    def model(self, cfg):
        return build_training_model(
            self.backbone(),
            cfg,
            self.graph()["cell"][0],
            device=torch.device("cpu"),
            dtype=DTYPE,
        )

    def test_yaml_force_training_checkpoint_and_ase(self):
        torch.manual_seed(743)
        for architecture, training in product(
            ("linear", "nonlinear"), ("frozen", "joint")
        ):
            with (
                self.subTest(architecture=architecture, training=training),
                tempfile.TemporaryDirectory() as tmp,
            ):
                cfg = self.configuration(
                    Path(tmp) / "train.yaml",
                    architecture,
                    training,
                    random_residual_initialization=True,
                )
                model = self.model(cfg)
                out = model(self.graph(), compute_force=True, training=True)
                (
                    out["energy"].square().mean() + out["forces"].square().mean()
                ).backward()
                self.assertTrue(
                    all(
                        p.grad is not None and torch.isfinite(p.grad).all()
                        for p in model.parameters()
                        if p.requires_grad
                    )
                )
                spectral = model.long_range.field_operator.fno.blocks[
                    0
                ].spectral.spectral
                for weight in spectral.parameters():
                    self.assertGreater(weight.grad.abs().max().item(), 0)
                delta = 1e-5
                energies = []
                for sign in (-1, 1):
                    graph = self.graph()
                    graph["positions"][0, 0] += sign * delta
                    energies.append(model(graph, compute_force=False)["energy"].sum())
                torch.testing.assert_close(
                    -(energies[1] - energies[0]) / (2 * delta),
                    out["forces"][0, 0],
                    atol=1e-7,
                    rtol=1e-5,
                )
                before = model.backbone.mace_model.local_scale.detach().clone()
                torch.optim.Adam(
                    [p for p in model.parameters() if p.requires_grad], lr=1e-3
                ).step()
                self.assertEqual(
                    torch.equal(before, model.backbone.mace_model.local_scale),
                    training == "frozen",
                )
                prepared = PreparedData(
                    samples=[],
                    train_samples=[],
                    validation_samples=[],
                    test_samples=[],
                    reference_cell=self.graph()["cell"][0],
                    train_cache_metadata={},
                    validation_cache_metadata=None,
                    test_cache_metadata=None,
                    train_cache_hit=False,
                    validation_cache_hit=False,
                    test_cache_hit=False,
                )
                result = OptimizationResult(
                    best_step=1,
                    best_validation_objective=0.0,
                    completed_steps=1,
                    stopped_early=False,
                    warmup_learning_rate=0,
                    final_learning_rate=1e-3,
                )
                payload = training_checkpoint_payload(
                    cfg, prepared, result, model, effective_configuration={}
                )
                self.assertEqual(payload["z_mixing"], "full")
                self.assertEqual(payload["spectral_rank"], cfg.model.spectral_rank)
                self.assertEqual(infer_checkpoint_z_mixing(payload), "full")
                restored = build_mace_fno_model(payload, self.backbone(), dtype=DTYPE)
                expected = model(self.graph(), compute_force=True)
                actual = restored(self.graph(), compute_force=True)
                for key in ("energy", "forces"):
                    torch.testing.assert_close(actual[key], expected[key])

                def converter(atoms):
                    graph = self.graph()
                    graph["positions"] = torch.tensor(atoms.positions, dtype=DTYPE)
                    graph["cell"] = torch.tensor(atoms.cell.array, dtype=DTYPE)[None]
                    graph["batch"] = torch.zeros(3, dtype=torch.long)
                    graph["ptr"] = torch.tensor([0, 3])
                    graph["node_attrs"] = graph["node_attrs"][:3]
                    return graph

                atoms = Atoms(
                    "H3",
                    positions=self.graph()["positions"][:3].numpy(),
                    cell=self.graph()["cell"][0].numpy(),
                    pbc=[True, True, False],
                )
                atoms.calc = MACEFNOCalculator(
                    model=restored, graph_converter=converter
                )
                direct = restored(converter(atoms), compute_force=True)
                self.assertAlmostEqual(
                    atoms.get_potential_energy(), direct["energy"].item(), places=12
                )
                torch.testing.assert_close(
                    torch.tensor(atoms.get_forces()), direct["forces"]
                )

    def test_zero_initialization_leaves_trainable_projection_and_warmup(self):
        for architecture in ("linear", "nonlinear"):
            with (
                self.subTest(architecture=architecture),
                tempfile.TemporaryDirectory() as tmp,
            ):
                cfg = self.configuration(
                    Path(tmp) / "train.yaml",
                    architecture,
                    output_initialization_scale=0.1,
                )
                self.configuration(
                    Path(tmp) / "warmup.yaml", architecture, output_warmup_steps=2
                )
                model = self.model(cfg)
                projection = model.long_range.field_operator.fno.projection_output
                initial = projection.weight.detach().clone()
                initialize_scaled_residual_output(model, 0.1)
                torch.testing.assert_close(projection.weight, 0.1 * initial)
                initialize_zero_residual(model)
                out = model(
                    self.graph(),
                    compute_force=True,
                    compute_base_force=True,
                    training=True,
                )
                torch.testing.assert_close(out["energy"], out["base_energy"])
                torch.testing.assert_close(out["forces"], out["base_forces"])
                parameters, active = configure_output_projection_warmup(model, 2)
                self.assertEqual(
                    {id(p) for p in active},
                    {id(p) for p in model.parameters() if p.requires_grad},
                )
                model.zero_grad()
                out["energy"].sum().backward()
                self.assertGreater(projection.weight.grad.abs().max().item(), 0)
                torch.optim.SGD(active, lr=0.1).step()
                finish_output_projection_warmup(parameters)
                model.zero_grad()
                model(self.graph(), compute_force=True, training=True)[
                    "forces"
                ].square().sum().backward()
                spectral = model.long_range.field_operator.fno.blocks[
                    0
                ].spectral.spectral
                for weight in spectral.parameters():
                    self.assertGreater(weight.grad.abs().max().item(), 0)

    def test_groups_apply_to_hidden_channels_and_legacy_linear_stays_direct(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.yaml"
            for architecture in ("linear", "nonlinear"):
                cfg = self.configuration(
                    path, architecture, channels=3, spectral_groups=2
                )
                self.model(cfg)
                with self.assertRaisesRegex(ValueError, "divisible"):
                    self.configuration(
                        path,
                        architecture,
                        channels=4,
                        fno_hidden_channels=3,
                        spectral_groups=2,
                    )
                for scheme in ("2d", "3d"):
                    with self.assertRaises(ValueError):
                        self.configuration(path, architecture, spatial_scheme=scheme)
            cfg = self.configuration(path, "linear", z_mixing="global")
            legacy = self.model(cfg).long_range.field_operator
            self.assertEqual(legacy.z_mixing, "spectral")
            self.assertFalse(hasattr(legacy.fno, "blocks"))
            self.assertEqual(
                infer_checkpoint_z_mixing(
                    {"architecture": "linear", "z_mixing": "spectral"}
                ),
                "local",
            )


if __name__ == "__main__":
    unittest.main()
