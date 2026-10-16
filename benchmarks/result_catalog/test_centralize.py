"""Boundary checks for the read-only catalog and its completion rules."""

import json
import tempfile
import unittest
from pathlib import Path

import centralize as c


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "runs"
        self.root.mkdir()
        self.output = self.root / "results"

    def run_dir(self, name, state=None, **extra):
        p = self.root / "_campaigns" / "au2_mgo_test_20261015" / "runs" / name
        p.mkdir(parents=True)
        if state:
            (p / "status.json").write_text(json.dumps(dict(state=state, **extra)))
        return p

    def test_excludes_active_failed_and_smoke_even_with_old_final_log(self):
        for name, state in [
            ("finished", "complete"),
            ("active", "running"),
            ("failed", "failed"),
            ("smoke_seed17", "complete"),
        ]:
            p = self.run_dir(name, state)
            (p / "model.pt").write_text("checkpoint placeholder")
            (p / "training.log").write_text("INFO: Done\n")
        self.run_dir("queued", "complete", job_id="123", array_task="2")
        rows, excluded, errors = c.discover(self.root, self.output, [dict(id="123_2")])
        self.assertEqual([r["run"] for r in rows], ["finished"])
        self.assertFalse(errors)
        self.assertEqual(len(excluded), 3)

    def test_legacy_finalization_requires_configured_budget(self):
        for name, step in [("complete_legacy", 20), ("partial_legacy", 10)]:
            p = self.run_dir(name)
            (p / "config.yaml").write_text("steps: 20\nfno_layers: 4\n")
            (p / "model.pt").write_text("checkpoint placeholder")
            (p / "train.log").write_text(
                f"step {step}/{step}: loss=1\ntimings: final-evaluation=2s, total=10s\n"
            )
        rows, _, _ = c.discover(self.root, self.output, [])
        self.assertEqual([r["run"] for r in rows], ["complete_legacy"])

    def test_terminal_run_overrides_obsolete_resume_stage_and_accepts_early_stop(self):
        p = self.run_dir("resumed", "complete")
        (p / "resume_123").mkdir()
        (p / "resume_123/status.json").write_text('{"state":"running"}')
        p = self.run_dir("early_stop")
        (p / "model.pt").write_text("checkpoint placeholder")
        (p / "config.yaml").write_text("steps: 20\nfno_layers: 4\n")
        (p / "train.log").write_text(
            "step 10/20: loss=1\nearly stopping at step 10\ntimings: final-evaluation=2s, total=10s\n"
        )
        rows, _, _ = c.discover(self.root, self.output, [])
        self.assertEqual({r["run"] for r in rows}, {"resumed", "early_stop"})

    def test_refresh_preserves_originals_and_previous_snapshot(self):
        p = self.run_dir("finished", "complete")
        (p / "test.json").write_text('{"energy_rmse": 1.0}')
        before = {str(f): f.read_bytes() for f in p.iterdir()}
        rows, excluded, errors = c.discover(self.root, self.output, [])
        first = c.build(self.root, self.output, rows, excluded, errors, [])
        second = c.build(self.root, self.output, rows, excluded, errors, [])
        self.assertNotEqual(first, second)
        self.assertEqual((self.output / "current").resolve(), second)
        self.assertTrue((first / "manifest.json").exists())
        self.assertEqual(before, {str(f): f.read_bytes() for f in p.iterdir()})
        self.assertEqual((second / rows[0]["directory"] / "files").resolve(), p)
        again, _, _ = c.discover(self.root, self.output, [])
        self.assertEqual(len(again), 1)


if __name__ == "__main__":
    unittest.main()
