"""Run from meanflow/: python -m unittest test_run_experiment. CLI checks."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import run_experiment
import workflow
from test_workflow import TINY, tiny_pixels

torch.set_num_threads(2)


def evaluate_args(root, **overrides):
    base = dict(device="cpu", run_dir=str(root / "run"), data_root=str(root / "data"),
                evaluation_name="fid_test", steps=[1, 2], num_samples=8,
                batch_size=4, seed=7, min_train_steps=2, max_minutes=60,
                save_every_images=8, checkpoint="")
    base.update(overrides)
    return SimpleNamespace(**base)


class Cli(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.pixels = tiny_pixels()
        workflow.train({**TINY}, run_dir=self.root / "run",
                       data_root=self.root / "data", target_steps=2,
                       session_steps=2, device="cpu", pixels=self.pixels,
                       save_every=2, log_every=2)
        self.calls = []

        def fake_evaluate(**kwargs):
            self.calls.append(kwargs)
            return {"completed": True, "results": {}}

        for target, attribute, value in [
            (run_experiment, "evaluate_fid", fake_evaluate),
        ]:
            patcher = patch.object(target, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_first_evaluate_pins_a_frozen_checkpoint(self):
        run_experiment.cmd_evaluate(evaluate_args(self.root))
        selection = json.loads(
            (self.root / "run" / "evaluations" / "fid_test" / "selection.json").read_text())
        self.assertEqual(selection["train_step"], 2)
        self.assertTrue(Path(selection["checkpoint"]).exists())
        self.assertNotIn("checkpoint-last", selection["checkpoint"])

    def test_rerun_reuses_the_pinned_checkpoint_after_more_training(self):
        run_experiment.cmd_evaluate(evaluate_args(self.root))
        first = json.loads(
            (self.root / "run" / "evaluations" / "fid_test" / "selection.json").read_text())
        workflow.train({**TINY}, run_dir=self.root / "run",
                       data_root=self.root / "data", target_steps=4,
                       session_steps=4, device="cpu", pixels=self.pixels,
                       save_every=2, log_every=2)
        run_experiment.cmd_evaluate(evaluate_args(self.root))
        second = json.loads(
            (self.root / "run" / "evaluations" / "fid_test" / "selection.json").read_text())
        self.assertEqual(first["checkpoint"], second["checkpoint"])
        self.assertEqual(second["train_step"], 2)

    def test_changed_fid_settings_need_a_new_evaluation_name(self):
        run_experiment.cmd_evaluate(evaluate_args(self.root))
        with self.assertRaises(SystemExit):
            run_experiment.cmd_evaluate(evaluate_args(self.root, num_samples=16))

    def test_explicit_checkpoint_on_an_existing_name_is_refused(self):
        run_experiment.cmd_evaluate(evaluate_args(self.root))
        with self.assertRaises(SystemExit):
            run_experiment.cmd_evaluate(evaluate_args(
                self.root, checkpoint=str(self.root / "run" / "checkpoint-last.pt")))

    def test_min_train_steps_blocks_an_undertrained_checkpoint(self):
        with self.assertRaises(SystemExit) as context:
            run_experiment.cmd_evaluate(evaluate_args(self.root, min_train_steps=1000))
        self.assertIn("min-train-steps", str(context.exception))

    def test_report_writes_a_row_per_requested_nfe(self):
        run_experiment.cmd_evaluate(evaluate_args(self.root))
        evaluation_dir = self.root / "run" / "evaluations" / "fid_test"
        (evaluation_dir / "results").mkdir(parents=True, exist_ok=True)
        (evaluation_dir / "results" / "fid_results.json").write_text(json.dumps(
            {"1": {"fid": 12.5, "n_fake": 8, "n_real": 8, "checkpoint_step": 2,
                   "seconds": 1.0}}))
        run_experiment.cmd_report(SimpleNamespace(
            run_dir=str(self.root / "run"), evaluation_name="fid_test"))
        rows = (evaluation_dir / "fid_comparison.csv").read_text().splitlines()
        self.assertEqual(len(rows), 3)          # header + NFE 1 and 2
        self.assertTrue(rows[1].startswith("1,12.5"))
        self.assertEqual(rows[2], "2,,,,,")     # NFE 2 is still blank
        self.assertTrue((evaluation_dir / "fid_vs_steps.png").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
