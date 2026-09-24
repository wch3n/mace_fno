"""Checkpoint-initialized fine-tuning without weakening exact-resume checks."""

from __future__ import annotations

import hashlib
import io
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch
import yaml
from mace_fno_test_helpers import FakeMACE, batch_data, train_arguments
from test_trainer import _sample, _ToyJointModel, _ToyResidual

from mace_fno import MACEFNOResidual
from mace_fno.cli import train_residual
from mace_fno.cli.config import parse_arguments
from mace_fno.cli.yaml_config import resolved_configuration
from mace_fno.training import (
    OptimizationResult,
    PreparedData,
    TrainingConfig,
    training_checkpoint_payload,
)
from mace_fno.training.checkpoint import mace_state_dict, residual_state_dict
from mace_fno.training.finetune import (
    initialize_finetune_model,
    load_finetune_checkpoint,
)
from mace_fno.training.resume import (
    RESUME_FORMAT_VERSION,
    resume_configuration,
    validate_resume_checkpoint,
)


class FineTuneFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.mace_path = self.root / "mace.model"
        self.mace_path.write_bytes(b"original MACE reference")
        self.data_path = self.root / "train.xyz"
        self.data_path.write_bytes(b"training labels")
        self.cell = torch.diag(torch.tensor([9.0, 10.0, 18.0], dtype=torch.float64))

    def arguments(self, *options):
        return train_arguments(
            "--mace-model",
            str(self.mace_path),
            "--train-file",
            str(self.data_path),
            "--grid",
            "4",
            "--z-grid",
            "4",
            "--modes",
            "1",
            "--spatial-scheme",
            "3d",
            "--spectral-symmetry",
            "metric_eqgino",
            "--channels",
            "2",
            "--source-hidden-channels",
            "4",
            "--fno-hidden-channels",
            "4",
            "--fno-layers",
            "1",
            "--dtype",
            "float64",
            "--device",
            "cpu",
            "--steps",
            "4",
            "--eval-interval",
            "1",
            "--learning-rate",
            "0.1",
            *options,
        )

    def configuration(self, *options):
        return TrainingConfig.from_namespace(self.arguments(*options))

    def model(self, regime="frozen"):
        model = MACEFNOResidual(
            FakeMACE(),
            (4, 4),
            2,
            (1, 1),
            invariant_indices=[0, 2],
            source_hidden_channels=4,
            fno_hidden_channels=4,
            fno_layers=1,
            spatial_scheme="3d",
            z_grid_size=4,
            reference_cell=self.cell,
            fno_spectral_symmetry="metric_eqgino",
            mace_training=regime,
        ).double()
        if regime == "joint":
            with torch.no_grad():
                model.backbone.mace_model.local_scale.fill_(0.12)
        return model

    def parent(self, *, latest=False, regime="frozen", fingerprint=True):
        args = self.arguments("--mace-training", regime)
        config = TrainingConfig.from_namespace(args)
        model = self.model(regime)
        if latest:
            payload = {
                "resume_format_version": RESUME_FORMAT_VERSION,
                "configuration": resume_configuration(config),
                "residual_state_dict": residual_state_dict(model),
                "mace_state_dict": mace_state_dict(model)
                if regime == "joint"
                else None,
                "completed_steps": 123,
                "best_step": 100,
                # Deliberately different best weights: latest initialization must
                # not accidentally load a selected-history snapshot.
                "best_residual_state_dict": {},
                "optimizer_state_dict": {"old": "must not be restored"},
                "stopped_early": True,
            }
        else:
            payload = training_checkpoint_payload(
                config,
                SimpleNamespace(reference_cell=self.cell),
                OptimizationResult(100, 0.01, 123, False, 0.1, 0.02),
                model,
                effective_configuration=resolved_configuration(args),
            )
        if fingerprint:
            payload["input_fingerprints"] = {
                "mace_model": hashlib.sha256(self.mace_path.read_bytes()).hexdigest()
            }
        path = self.root / ("parent.last.pt" if latest else "parent.energy.pt")
        torch.save(payload, path)
        return path, payload, model


class FineTuneTests(FineTuneFixtures):
    def test_yaml_init_from_and_changed_scales(self):
        path = self.root / "fine.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "mace_model": "mace.model",
                    "train_file": "train.xyz",
                    "init_from": "old/model.energy.pt",
                    "checkpoint": "new/model.pt",
                    "energy_scale": 0.3,
                    "force_scale": 2.0,
                }
            )
        )
        config = TrainingConfig.from_namespace(parse_arguments(["--config", str(path)]))
        self.assertEqual(config.runtime.init_from, self.root / "old/model.energy.pt")
        self.assertEqual(config.optimization.energy_scale, 0.3)
        self.assertEqual(config.optimization.force_scale, 2.0)
        self.assertIsNone(config.runtime.resume)

    def test_parent_and_resume_collision_guards(self):
        parent = self.root / "parent.pt"
        cases = (
            ("--resume", "other.last.pt"),
            ("--checkpoint", str(parent)),
            ("--last-checkpoint", str(parent)),
            ("--train-cache", str(parent)),
            ("--validation-cache", str(parent)),
            ("--test-cache", str(parent)),
            ("--spectral-diagnostic-output", str(parent)),
        )
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.configuration("--init-from", str(parent), *options)
        for parent_name in ("model.energy.pt", "model.last.pt", "model.config.yaml"):
            with (
                self.subTest(parent=parent_name),
                self.assertRaisesRegex(ValueError, "overwrite"),
            ):
                self.configuration(
                    "--init-from",
                    str(self.root / parent_name),
                    "--checkpoint",
                    str(self.root / "model.pt"),
                    "--energy-checkpoint-tolerance",
                    "0.01",
                )
        parent.write_bytes(b"parent")
        alias = self.root / "alias.pt"
        alias.symlink_to(parent)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.configuration("--init-from", str(parent), "--checkpoint", str(alias))

    def test_frozen_and_joint_weights_and_predictions_match(self):
        for regime in ("frozen", "joint"):
            for latest in (False, True):
                with self.subTest(regime=regime, latest=latest):
                    path, _, original = self.parent(latest=latest, regime=regime)
                    config = self.configuration(
                        "--mace-training",
                        regime,
                        "--energy-scale",
                        "0.3",
                        "--force-scale",
                        "2",
                        "--learning-rate",
                        "0.0001",
                    )
                    loaded = load_finetune_checkpoint(path, config)
                    target = self.model(regime)
                    if regime == "joint":
                        with torch.no_grad():
                            target.backbone.mace_model.local_scale.fill_(0.07)
                    initialize_finetune_model(target, loaded)
                    for key, value in original.state_dict().items():
                        self.assertTrue(
                            torch.equal(value, target.state_dict()[key]), key
                        )
                    for model in (original, target):
                        model.eval()
                    before = original(
                        batch_data(), training=False, compute_residual_force=False
                    )
                    after = target(
                        batch_data(), training=False, compute_residual_force=False
                    )
                    torch.testing.assert_close(
                        before["energy"], after["energy"], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        before["forces"], after["forces"], rtol=0, atol=0
                    )
                    self.assertEqual(
                        loaded.provenance["parent_step"], 123 if latest else 100
                    )
                    self.assertEqual(loaded.provenance["optimizer_state"], "reset")

    def test_resolved_z_mode_default_is_compatible(self):
        path, payload, _ = self.parent()
        payload["training_configuration"]["z_modes"] = 1
        torch.save(payload, path)
        load_finetune_checkpoint(path, self.configuration())

    def test_incompatible_model_precision_mode_head_and_backbone_are_rejected(self):
        path, _, _ = self.parent()
        for option in (
            ("--grid", "6"),
            ("--channels", "4"),
            ("--fno-layers", "2"),
            ("--dtype", "float32"),
            ("--mace-training", "joint"),
            ("--head", "other"),
        ):
            with self.subTest(option=option), self.assertRaises(ValueError):
                load_finetune_checkpoint(path, self.configuration(*option))
        self.mace_path.write_bytes(b"different MACE with the same name")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_finetune_checkpoint(path, self.configuration())

    def test_legacy_reference_fallback_and_relocated_hashed_backbone(self):
        path, _, _ = self.parent(fingerprint=False)
        loaded = load_finetune_checkpoint(path, self.configuration())
        self.assertEqual(
            loaded.provenance["mace_reference_verification"],
            "original_reference_file_sha256",
        )
        self.mace_path.rename(self.root / "relocated.model")
        changed = self.configuration("--mace-model", str(self.root / "relocated.model"))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            load_finetune_checkpoint(path, changed)
        payload = torch.load(path, weights_only=False)
        payload["input_fingerprints"] = {
            "mace_model": hashlib.sha256(b"original MACE reference").hexdigest()
        }
        torch.save(payload, path)
        load_finetune_checkpoint(path, changed)

    def test_missing_or_wrong_shaped_weights_fail_without_mutating_model(self):
        path, _, _ = self.parent(regime="joint")
        loaded = load_finetune_checkpoint(
            path, self.configuration("--mace-training", "joint")
        )
        target = self.model("joint")
        before = deepcopy(target.state_dict())
        key = next(k for k in loaded.residual_state if k.startswith("source_head"))
        missing = dict(loaded.residual_state)
        missing.pop(key)
        bad_mace = dict(loaded.mace_state)
        bad_mace["local_scale"] = torch.ones(2)
        for damaged in (
            replace(loaded, residual_state=missing),
            replace(loaded, mace_state=bad_mace),
        ):
            with self.assertRaisesRegex(ValueError, "keys differ|shape differs"):
                initialize_finetune_model(target, damaged)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, target.state_dict()[key]))

    def test_exact_resume_still_rejects_changed_scales(self):
        path, payload, _ = self.parent(latest=True)
        self.assertTrue(path.exists())
        with self.assertRaisesRegex(ValueError, "settings differ"):
            validate_resume_checkpoint(
                payload, self.configuration("--energy-scale", "0.3")
            )


class _ReferencedResidual(_ToyResidual):
    def __init__(self, cell):
        super().__init__()
        self.register_buffer("reference_cell", cell.clone())

    def _validate_cells(self, cells):
        del cells


class _ReferencedJoint(_ToyJointModel):
    def __init__(self, cell):
        super().__init__()
        self.register_buffer("reference_cell", cell.clone())

    def _validate_cells(self, cells):
        del cells


class FineTuneCLITests(FineTuneFixtures):
    def test_cli_fresh_state_parent_provenance_and_subsequent_resume(self):
        for regime in ("frozen", "joint"):
            for source_kind in ("model.pt", "model.energy.pt", "model.last.pt"):
                with self.subTest(regime=regime, source=source_kind):
                    directory = self.root / regime / source_kind
                    sample = _sample(0.5)
                    prepared = PreparedData(
                        samples=[sample],
                        train_samples=[sample],
                        validation_samples=[sample],
                        test_samples=[],
                        reference_cell=self.cell,
                        train_cache_metadata={},
                        validation_cache_metadata=None,
                        test_cache_metadata=None,
                        train_cache_hit=True,
                        validation_cache_hit=True,
                        test_cache_hit=False,
                    )
                    states = []
                    built_references = []
                    cache_backbones = []
                    real_save = train_residual.save_training_checkpoint

                    def save(path, payload):
                        if "resume_format_version" in payload:
                            states.append(deepcopy(payload))
                        return real_save(path, payload)

                    def build(_backbone, _configuration, cell, **_kwargs):
                        built_references.append(cell.clone())
                        factory = (
                            _ReferencedJoint
                            if regime == "joint"
                            else _ReferencedResidual
                        )
                        return factory(cell)

                    def cache(model, *_args, **_kwargs):
                        if regime == "joint":
                            cache_backbones.append(
                                float(model.backbone.mace_model.scale.detach())
                            )

                    def invoke(output, steps, *options):
                        args = self.arguments(
                            "--mace-training",
                            regime,
                            "--checkpoint",
                            str(output),
                            "--steps",
                            str(steps),
                            "--energy-checkpoint-tolerance",
                            "0.01",
                            *options,
                        )
                        with (
                            redirect_stdout(io.StringIO()),
                            mock.patch.object(
                                train_residual, "parse_arguments", return_value=args
                            ),
                            mock.patch.object(
                                train_residual,
                                "load_mace_calculator",
                                return_value=SimpleNamespace(models=[None]),
                            ),
                            mock.patch.object(
                                train_residual,
                                "prepare_data",
                                return_value=deepcopy(prepared),
                            ),
                            mock.patch.object(
                                train_residual,
                                "build_training_model",
                                side_effect=build,
                            ),
                            mock.patch.object(
                                train_residual,
                                "cache_frozen_targets",
                                side_effect=cache,
                            ),
                            mock.patch.object(
                                train_residual,
                                "save_training_checkpoint",
                                side_effect=save,
                            ),
                        ):
                            train_residual.main()

                    original = directory / "parent/model.pt"
                    invoke(original, 4)
                    parent = original.with_name(source_kind)
                    parent_bytes = parent.read_bytes()
                    payload = torch.load(parent, weights_only=False)
                    states.clear()
                    # Exercise preservation of the spectral anchor cell when the
                    # new data loader chooses a different first/reference cell.
                    prepared.reference_cell = self.cell * 1.2
                    output = directory / "fine/model.pt"
                    options = (
                        "--energy-scale",
                        "0.3",
                        "--force-scale",
                        "2",
                        "--learning-rate",
                        "0.02",
                    )
                    invoke(output, 2, "--init-from", str(parent), *options)
                    first = states[0]
                    self.assertEqual(first["completed_steps"], 0)
                    self.assertEqual(first["best_step"], 0)
                    self.assertEqual(first["optimizer_state_dict"]["state"], {})
                    self.assertEqual(
                        first["optimizer_state_dict"]["param_groups"][0]["lr"], 0.02
                    )
                    self.assertFalse(first["stopped_early"])
                    self.assertEqual(first["spectral_history"], [])
                    for key, value in payload["residual_state_dict"].items():
                        self.assertTrue(
                            torch.equal(value, first["residual_state_dict"][key])
                        )
                    if regime == "joint":
                        for key, value in payload["mace_state_dict"].items():
                            self.assertTrue(
                                torch.equal(value, first["mace_state_dict"][key])
                            )
                        self.assertEqual(cache_backbones, [0.0, 0.0])
                    self.assertTrue(torch.equal(built_references[-1], self.cell))
                    self.assertEqual(parent.read_bytes(), parent_bytes)
                    selected = torch.load(output, weights_only=False)
                    lineage = selected["fine_tuning"]
                    self.assertEqual(
                        lineage["parent_sha256"],
                        hashlib.sha256(parent_bytes).hexdigest(),
                    )
                    self.assertTrue(torch.equal(selected["reference_cell"], self.cell))
                    self.assertIn("input_fingerprints", selected)
                    latest = output.with_name("model.last.pt")
                    self.assertEqual(
                        torch.load(latest, weights_only=False)["completed_steps"], 2
                    )
                    parent.unlink()  # Exact resume must not need the original parent.
                    continued = directory / "continued/model.pt"
                    invoke(continued, 4, "--resume", str(latest), *options)
                    final = torch.load(
                        continued.with_name("model.last.pt"), weights_only=False
                    )
                    self.assertEqual(final["fine_tuning"], lineage)
                    self.assertEqual(final["completed_steps"], 4)


if __name__ == "__main__":
    unittest.main()
