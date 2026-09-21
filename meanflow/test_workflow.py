"""Run from meanflow/: python -m unittest test_workflow. CPU pipeline checks.

The FID numbers produced here use a tiny stand-in feature extractor so the
tests stay offline and fast. They are not research results.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import workflow
from workflow import (FeatureStats, evaluate_fid, frechet_distance,
                      freeze_checkpoint, initial_noise, load_checkpoint,
                      sampling_cost, train, validate_config)

torch.set_num_threads(2)

TINY = {
    "method": "weak", "model_channels": 8, "batch_size": 4, "seed": 0,
    "lr": 1e-4, "warmup_steps": 2, "ema_decay": 0.99, "horizontal_flip": True,
    "weak_features": 8, "weak_test_family": "endpoint",
    "weak_time_sampler": "logitnormal",
}

FEATURES = 6


class StubInception(torch.nn.Module):
    """Deterministic low-dimensional features; offline stand-in for Inception."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.register_buffer("projection", torch.randn(3 * 8 * 8, FEATURES,
                                                       generator=generator))

    def forward(self, images):
        flat = images.float().flatten(1) / 255.0
        return flat @ self.projection


def tiny_pixels(count=64):
    generator = torch.Generator().manual_seed(1)
    return (torch.rand(count, 3, 8, 8, generator=generator) * 255).to(torch.uint8)


class Config(unittest.TestCase):
    def test_weak_config_rejects_diag_probability(self):
        with self.assertRaises(ValueError) as context:
            validate_config({**TINY, "diag_probability": 0.25})
        self.assertIn("no effect when method=weak", str(context.exception))

    def test_baseline_config_requires_diag_probability(self):
        with self.assertRaises(ValueError):
            validate_config({"method": "mf_control", "model_channels": 8})
        ok = validate_config({"method": "mf_control", "model_channels": 8,
                              "diag_probability": 0.25})
        self.assertEqual(ok["method"], "mf_control")

    def test_baseline_config_rejects_weak_only_keys(self):
        with self.assertRaises(ValueError) as context:
            validate_config({"method": "imf_diag", "diag_probability": 0.25,
                             "weak_features": 32})
        self.assertIn("weak-only keys", str(context.exception))

    def test_merged_config_round_trips_for_every_method(self):
        """A checkpoint stores the merged config; re-validating it must pass."""
        for config in ({"method": "mf_control", "diag_probability": 0.25},
                       {"method": "mf"}, {"method": "weak"}):
            merged = validate_config(config)
            self.assertEqual(validate_config(merged), merged)

    def test_mf_config_uses_ratio_not_diag_probability(self):
        ok = validate_config({"method": "mf", "ratio": 0.5, "norm_p": 1.0})
        self.assertEqual((ok["ratio"], ok["norm_p"], ok["tr_sampler"]), (0.5, 1.0, "v0"))
        with self.assertRaises(ValueError):
            validate_config({"method": "mf", "diag_probability": 0.25})
        with self.assertRaises(ValueError) as context:
            validate_config({"method": "weak", "ratio": 0.5})
        self.assertIn("mf-only keys", str(context.exception))

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({**TINY, "lerning_rate": 1e-4})

    def test_endpoint_with_adjoint_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({**TINY, "weak_time_correction": "adjoint"})


class Statistics(unittest.TestCase):
    def test_moments_match_a_direct_computation(self):
        torch.manual_seed(2)
        data = torch.randn(200, FEATURES, dtype=torch.float64)
        stats = FeatureStats(FEATURES)
        for chunk in data.split(37):
            stats.update(chunk)
        mean, covariance = stats.moments()
        torch.testing.assert_close(mean, data.mean(0))
        torch.testing.assert_close(covariance, data.T.cov())

    def test_identical_distributions_give_zero_distance(self):
        torch.manual_seed(3)
        data = torch.randn(300, FEATURES, dtype=torch.float64)
        a, b = FeatureStats(FEATURES), FeatureStats(FEATURES)
        a.update(data)
        b.update(data)
        self.assertAlmostEqual(frechet_distance(a, b), 0.0, places=6)

    def test_shifted_distribution_gives_the_squared_shift(self):
        torch.manual_seed(4)
        data = torch.randn(4000, FEATURES, dtype=torch.float64)
        shift = torch.full((FEATURES,), 0.5, dtype=torch.float64)
        a, b = FeatureStats(FEATURES), FeatureStats(FEATURES)
        a.update(data)
        b.update(data + shift)
        self.assertAlmostEqual(frechet_distance(a, b),
                               float(shift.dot(shift)), places=4)

    def test_state_round_trip(self):
        torch.manual_seed(5)
        stats = FeatureStats(FEATURES)
        stats.update(torch.randn(50, FEATURES, dtype=torch.float64))
        restored = FeatureStats.from_state(stats.state())
        self.assertEqual(restored.count, stats.count)
        torch.testing.assert_close(restored.outer, stats.outer)


class Noise(unittest.TestCase):
    def test_initial_noise_is_reproducible_and_batch_dependent(self):
        shape = (4, 3, 8, 8)
        a = initial_noise(12345, 2, shape, "cpu")
        b = initial_noise(12345, 2, shape, "cpu")
        c = initial_noise(12345, 3, shape, "cpu")
        torch.testing.assert_close(a, b)
        self.assertFalse(torch.allclose(a, c))


class Cost(unittest.TestCase):
    def test_sampling_cost_arithmetic(self):
        cost = sampling_cost([1, 2, 4, 8, 16, 32, 64, 128], 50_000)
        self.assertEqual(cost["image_nfe_total"], 12_750_000)
        self.assertEqual(cost["inception_images"], 450_000)
        self.assertAlmostEqual(cost["share"][64] + cost["share"][128], 0.7529, places=3)


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.pixels = tiny_pixels()
        patcher = patch.object(workflow, "_inception", lambda device: StubInception())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.directory.cleanup)

    def _train(self, steps, **overrides):
        return train({**TINY, **overrides}, run_dir=self.root / "run",
                     data_root=self.root / "data", target_steps=steps,
                     session_steps=steps, device="cpu", pixels=self.pixels,
                     save_every=2, log_every=1)

    def test_training_resumes_from_the_last_save(self):
        first = self._train(4)
        self.assertEqual(first["step"], 4)
        step_after_first = load_checkpoint(self.root / "run" / "checkpoint-last.pt")["step"]
        second = self._train(8)
        self.assertEqual(step_after_first, 4)
        self.assertEqual(second["step"], 8)
        self.assertTrue(second["completed"])
        status = json.loads((self.root / "run" / "train_status.json").read_text())
        self.assertEqual(status["step"], 8)

    def test_changed_config_refuses_to_resume(self):
        self._train(2)
        with self.assertRaises(ValueError) as context:
            self._train(4, lr=5e-4)
        self.assertIn("changed", str(context.exception))

    def test_frozen_checkpoint_is_content_addressed(self):
        self._train(2)
        source = self.root / "run" / "checkpoint-last.pt"
        first = freeze_checkpoint(source, self.root / "frozen")
        second = freeze_checkpoint(source, self.root / "frozen")
        self.assertEqual(first, second)
        self.assertEqual(load_checkpoint(first)["step"], 2)

    def test_multi_nfe_fid_completes_and_resumes(self):
        self._train(2)
        frozen = freeze_checkpoint(self.root / "run" / "checkpoint-last.pt",
                                   self.root / "frozen")
        common = dict(checkpoint=frozen, output_dir=self.root / "fid",
                      data_root=self.root / "data", nfe_values=[1, 2, 4],
                      num_samples=16, batch_size=8, seed=7, device="cpu",
                      pixels=self.pixels, save_every_images=8)
        # A zero-minute budget stops after the first partial save.
        partial = evaluate_fid(max_minutes=0, **common)
        self.assertFalse(partial["completed"])
        finished = evaluate_fid(max_minutes=60, **common)
        self.assertTrue(finished["completed"])
        self.assertEqual(sorted(finished["results"]), ["1", "2", "4"])
        for entry in finished["results"].values():
            self.assertEqual(entry["n_fake"], 16)
            self.assertEqual(entry["n_real"], 16)
            self.assertTrue(entry["fid"] == entry["fid"])  # not NaN

    def test_completed_nfe_is_not_recomputed(self):
        self._train(2)
        frozen = freeze_checkpoint(self.root / "run" / "checkpoint-last.pt",
                                   self.root / "frozen")
        common = dict(checkpoint=frozen, output_dir=self.root / "fid",
                      data_root=self.root / "data", nfe_values=[1],
                      num_samples=16, batch_size=8, seed=7, device="cpu",
                      pixels=self.pixels, max_minutes=60)
        first = evaluate_fid(**common)
        second = evaluate_fid(**common)
        self.assertEqual(first["results"]["1"]["fid"], second["results"]["1"]["fid"])
        self.assertIn("seconds", second["results"]["1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
