"""Run from meanflow/: python -m unittest test_weak. CPU mathematical checks."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from models.meanflow import MeanFlow
from models.unet import SongUNet
from models.weak_loss import (fourier_coefficients, experimental_loss,
                              sample_time_pair, u_statistic, u_statistic_terms,
                              weak_terms, _weak_settings)

torch.set_num_threads(2)

D = 2
X0 = torch.tensor([0.3, -0.7], dtype=torch.float64)


def settings(method="weak", **overrides):
    base = dict(
        method=method, weak_features=32, weak_weight=1.0, diag_weight=1.0,
        weak_sigma_z=1.0, weak_sigma_r=1.0, weak_sigma_t=1.0,
        weak_fp64=False, ema_decay=0.999, ema_decays=[],
        weak_test_family="vanishing", weak_time_sampler="uniform",
        weak_time_correction="importance", weak_mixture_alpha=0.2,
        weak_P_mean_t=-0.6, weak_P_std_t=1.6, weak_P_mean_r=-4.0,
        weak_P_std_r=1.6, weak_diag_time="uniform", diag_probability=None,
    )
    if method != "weak":
        base["diag_probability"] = 0.25
    base.update(overrides)
    return SimpleNamespace(**base)


class ExactSolution:
    """u(z,r,t) = scale * (z - X0) / t is the exact mean flow for x == X0."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, z, tr, aug_cond):
        t = tr[0].view(-1, *([1] * (z.ndim - 1)))
        return self.scale * (z - X0.view(1, -1, 1, 1)) / t


class UStatistic(unittest.TestCase):
    def test_matrix_value_and_gradient_match_all_distinct_pairs(self):
        torch.manual_seed(10)
        B, M, d = 5, 7, 3
        F = torch.randn(B, d, dtype=torch.float64, requires_grad=True)
        v = torch.randn(B, d, dtype=torch.float64)
        A, phi = torch.randn(2, B, M, dtype=torch.float64)
        loss = u_statistic(F, v, A, phi)
        h = A[:, :, None] * F[:, None, :] + phi[:, :, None] * v[:, None, :]
        direct = sum((h[i] * h[j]).sum() for i in range(B) for j in range(B) if i != j)
        direct = direct / (B * (B - 1) * M * d)
        torch.testing.assert_close(loss, direct, atol=1e-12, rtol=1e-12)
        g1 = torch.autograd.grad(loss, F, retain_graph=True)[0]
        g2 = torch.autograd.grad(direct, F)[0]
        torch.testing.assert_close(g1, g2, atol=1e-12, rtol=1e-12)

    def test_three_term_form_matches_all_distinct_pairs(self):
        """The endpoint term adds a third (coefficient, vector) pair."""
        torch.manual_seed(7)
        B, M, d = 5, 4, 3
        Cs = [torch.randn(B, M, dtype=torch.float64) for _ in range(3)]
        Vs = [torch.randn(B, d, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        value = u_statistic_terms(Cs, Vs)
        h = sum(C[:, :, None] * V[:, None, :] for C, V in zip(Cs, Vs))
        direct = sum((h[i] * h[j]).sum() for i in range(B) for j in range(B) if i != j)
        direct = direct / (B * (B - 1) * M * d)
        torch.testing.assert_close(value, direct, atol=1e-12, rtol=1e-12)
        g1 = torch.autograd.grad(value, Vs[0], retain_graph=True)[0]
        g2 = torch.autograd.grad(direct, Vs[0])[0]
        torch.testing.assert_close(g1, g2, atol=1e-12, rtol=1e-12)

    def test_negative_estimate_is_preserved(self):
        F = torch.tensor([[1.0], [-1.0]], requires_grad=True)
        loss = u_statistic(F, torch.zeros_like(F), torch.ones(2, 1), torch.zeros(2, 1))
        self.assertEqual(loss.item(), -1.0)

    def test_one_sample_rejected(self):
        with self.assertRaises(ValueError):
            u_statistic(torch.zeros(1, 1), torch.zeros(1, 1),
                        torch.ones(1, 1), torch.zeros(1, 1))

    def test_bias_and_unbiasedness_with_known_distribution(self):
        torch.manual_seed(30)
        trials, B, d = 40000, 4, 2
        mu = torch.tensor([0.6, -0.4], dtype=torch.float64)
        h = mu + torch.randn(trials, B, d, dtype=torch.float64)
        estimates = (h.sum(1).square().sum(1) - h.square().sum((1, 2))) / (B * (B - 1) * d)
        target = mu.square().mean()
        error = (estimates.mean() - target).abs()
        self.assertLess(error.item(), 5 * estimates.std().item() / (trials ** 0.5))
        naive = h.mean(1).square().mean(1).mean()
        self.assertAlmostEqual((naive - target).item(), 1 / B, delta=0.015)


class TestFunctions(unittest.TestCase):
    def test_fourier_transport_derivative_by_finite_difference(self):
        torch.manual_seed(20)
        z, v = torch.randn(2, 4, 3, dtype=torch.float64)
        t = torch.full((4, 1), 0.7, dtype=torch.float64)
        r = torch.full((4, 1), 0.2, dtype=torch.float64)
        W = torch.randn(5, 3, dtype=torch.float64)
        b, c, phase = torch.randn(3, 5, dtype=torch.float64)
        for family in ("vanishing", "endpoint"):
            A, _ = fourier_coefficients(z, v, t, r, W, b, c, phase, family=family)
            eps = 1e-6
            _, plus = fourier_coefficients(z + eps * v, v, t + eps, r, W, b, c, phase,
                                           family=family)
            _, minus = fourier_coefficients(z - eps * v, v, t - eps, r, W, b, c, phase,
                                            family=family)
            torch.testing.assert_close(A, (plus - minus) / (2 * eps),
                                       atol=1e-8, rtol=1e-8, msg=family)

    def test_adjoint_term_is_phi_times_dlogq(self):
        torch.manual_seed(3)
        z, v = torch.randn(2, 6, 3, dtype=torch.float64)
        t = torch.rand(6, 1, dtype=torch.float64) * 0.5 + 0.25
        r = t * 0.4
        W = torch.randn(5, 3, dtype=torch.float64)
        b, c, phase = torch.randn(3, 5, dtype=torch.float64)
        dlogq = t.flatten().square() * 0.7
        A, phi = fourier_coefficients(z, v, t, r, W, b, c, phase,
                                      family="endpoint", dlogq_dt=dlogq)
        plain, _ = fourier_coefficients(z, v, t, r, W, b, c, phase, family="endpoint")
        torch.testing.assert_close(A - plain, phi * dlogq.reshape(-1, 1),
                                   atol=1e-12, rtol=1e-12)

    def test_true_constant_velocity_has_zero_weak_integral(self):
        # Independent quadrature checks signs, F=(t-r)u and the upper boundary.
        q, weights = np.polynomial.legendre.leggauss(80)
        t = torch.tensor((q + 1) / 2, dtype=torch.float64).reshape(-1, 1)
        w = torch.tensor(weights / 2, dtype=torch.float64).reshape(-1, 1)
        r = torch.zeros_like(t)
        v = torch.tensor([0.7, -0.2], dtype=torch.float64).expand(len(t), -1)
        z = torch.tensor([0.1, 0.3], dtype=torch.float64) + t * v
        F = t * v
        W = torch.tensor([[1.1, -0.8], [0.4, 1.4]], dtype=torch.float64)
        b = torch.tensor([0.3, -0.4], dtype=torch.float64)
        c = torch.tensor([0.8, 0.5], dtype=torch.float64)
        phase = torch.tensor([0.2, 1.1], dtype=torch.float64)
        A, phi = fourier_coefficients(z, v, t, r, W, b, c, phase)
        h = A[:, :, None] * F[:, None, :] + phi[:, :, None] * v[:, None, :]
        integral = (w[:, :, None] * h).sum(0)
        torch.testing.assert_close(integral, torch.zeros_like(integral),
                                   atol=1e-12, rtol=0)

    def test_endpoint_interior_integral_equals_boundary_term(self):
        """D.138: with phi = cos(xi) the interior integral equals B_phi(r)."""
        q, weights = np.polynomial.legendre.leggauss(120)
        for r_value in (0.0, 0.35):
            half = (1 - r_value) / 2
            t = torch.tensor(r_value + half * (q + 1), dtype=torch.float64).reshape(-1, 1)
            w = torch.tensor(weights * half, dtype=torch.float64).reshape(-1, 1)
            r = torch.full_like(t, r_value)
            v = torch.tensor([0.7, -0.2], dtype=torch.float64).expand(len(t), -1)
            z0 = torch.tensor([0.1, 0.3], dtype=torch.float64)
            z = z0 + t * v
            F = (t - r) * v
            W = torch.tensor([[1.1, -0.8], [0.4, 1.4]], dtype=torch.float64)
            b = torch.tensor([0.3, -0.4], dtype=torch.float64)
            c = torch.tensor([0.8, 0.5], dtype=torch.float64)
            phase = torch.tensor([0.2, 1.1], dtype=torch.float64)
            A, phi = fourier_coefficients(z, v, t, r, W, b, c, phase, family="endpoint")
            h = A[:, :, None] * F[:, None, :] + phi[:, :, None] * v[:, None, :]
            interior = (w[:, :, None] * h).sum(0)
            z1 = z0 + v[0]
            xi1 = z1 @ W.T + r_value * b + c + phase
            boundary = xi1.cos()[:, None] * ((1 - r_value) * v[0])[None, :]
            torch.testing.assert_close(interior, boundary, atol=1e-10, rtol=0)


class TimeSamplers(unittest.TestCase):
    def test_uniform_triangle_density(self):
        t, r, q, dlogq = sample_time_pair(20000, "cpu", torch.float64)
        self.assertTrue(bool((r < t).all()))
        torch.testing.assert_close(q, torch.full_like(q, 2.0))
        torch.testing.assert_close(dlogq, torch.zeros_like(dlogq))

    def test_mixture_density_normalises_and_is_bounded(self):
        for alpha in (0.2, 0.5):
            t, r, q, _ = sample_time_pair(200000, "cpu", torch.float64,
                                          sampler="logitnormal", mixture_alpha=alpha)
            self.assertTrue(bool((r < t).all()))
            # E_q[2/q] = 2 * area(T) = 1 confirms the density normalisation.
            self.assertAlmostEqual(float((2.0 / q).mean()), 1.0, delta=0.02)
            self.assertGreaterEqual(float(q.min()), 2 * alpha - 1e-9)
            self.assertLessEqual(float((2.0 / q).max()), 1.0 / alpha + 1e-9)

    def test_dlogq_matches_finite_difference_of_log_density(self):
        from models.weak_loss import (_logit_normal_log_density as logf,
                                      _logit_normal_dlog_density as dlogf)
        s = torch.tensor([0.05, 0.3, 0.6, 0.9], dtype=torch.float64)
        eps = 1e-6
        fd = (logf(s + eps, -0.6, 1.6) - logf(s - eps, -0.6, 1.6)) / (2 * eps)
        torch.testing.assert_close(dlogf(s, -0.6, 1.6), fd, atol=1e-6, rtol=1e-6)


class Estimator(unittest.TestCase):
    """The weak estimate must vanish at the exact solution for every variant."""

    VARIANTS = [
        dict(),
        dict(weak_test_family="endpoint"),
        dict(weak_time_sampler="logitnormal"),
        dict(weak_time_sampler="logitnormal", weak_time_correction="adjoint"),
        dict(weak_test_family="endpoint", weak_time_sampler="logitnormal"),
    ]

    def _mean(self, args, scale, trials=1500, batch=8):
        torch.manual_seed(0)
        net = ExactSolution(scale)
        values = []
        for _ in range(trials):
            x = X0.view(1, -1, 1, 1).expand(batch, D, 1, 1).contiguous()
            _, weak, _ = weak_terms(net, x, args)
            values.append(float(weak))
        tensor = torch.tensor(values, dtype=torch.float64)
        return float(tensor.mean()), float(tensor.std()) / math.sqrt(trials)

    def test_zero_at_the_exact_solution_and_positive_when_perturbed(self):
        for overrides in self.VARIANTS:
            with self.subTest(**overrides):
                args = settings(weak_fp64=True, **overrides)
                mean, se = self._mean(args, 1.0)
                self.assertLess(abs(mean), 4 * se + 1e-9,
                                f"not centred on zero: {mean:.3e} +- {se:.1e}")
                wrong, wrong_se = self._mean(args, 2.0)
                self.assertGreater(wrong, 4 * wrong_se,
                                   f"did not detect a wrong solution: {wrong:.3e}")


class Configuration(unittest.TestCase):
    def test_diag_probability_rejected_for_weak(self):
        with self.assertRaises(ValueError):
            _weak_settings(settings(diag_probability=0.25))

    def test_endpoint_with_adjoint_rejected(self):
        with self.assertRaises(ValueError):
            _weak_settings(settings(weak_test_family="endpoint",
                                    weak_time_sampler="logitnormal",
                                    weak_time_correction="adjoint"))

    def test_unknown_family_and_sampler_rejected(self):
        with self.assertRaises(ValueError):
            _weak_settings(settings(weak_test_family="nope"))
        with self.assertRaises(ValueError):
            sample_time_pair(4, "cpu", torch.float32, sampler="nope")


class RealModel(unittest.TestCase):
    CONFIG = dict(img_resolution=8, in_channels=3, out_channels=3,
                  model_channels=16, channel_mult=[1, 1], num_blocks=1,
                  attn_resolutions=[4], dropout=0)

    def test_official_unet_loss_backward_and_sampling(self):
        for method in ["weak", "mf_control", "imf_diag"]:
            with self.subTest(method=method):
                torch.manual_seed(40)
                model = MeanFlow(SongUNet, settings(method), self.CONFIG)
                x = torch.randn(4, 3, 8, 8)
                optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-4)
                if method == "weak":
                    with patch("torch.func.jvp",
                               side_effect=AssertionError("weak must not use JVP")):
                        loss, _ = experimental_loss(model.net, x, model.args)
                else:
                    loss = model.forward_with_loss(x, None)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                grads = [p.grad for p in model.net.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
                optimizer.step()
                model.update_ema()
                model.eval()
                with torch.no_grad():
                    sample = model.sample((2, 3, 8, 8), device="cpu", num_steps=3)
                self.assertEqual(sample.shape, (2, 3, 8, 8))
                self.assertTrue(torch.isfinite(sample).all())

    def test_split_backward_matches_single_backward(self):
        """diag.backward() + weak.backward() == (diag + weak).backward()."""
        def gradients(split):
            torch.manual_seed(41)
            model = MeanFlow(SongUNet, settings("weak"), self.CONFIG)
            x = torch.randn(4, 3, 8, 8)
            torch.manual_seed(99)
            diag, weak, _ = model.loss_terms(x)
            if split:
                diag.backward()
                weak.backward()
            else:
                (diag + weak).backward()
            return [p.grad.clone() for p in model.net.parameters() if p.grad is not None]

        for a, b in zip(gradients(True), gradients(False)):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_endpoint_variant_trains_on_the_real_unet(self):
        torch.manual_seed(42)
        args = settings("weak", weak_test_family="endpoint",
                        weak_time_sampler="logitnormal")
        model = MeanFlow(SongUNet, args, self.CONFIG)
        diag, weak, logs = model.loss_terms(torch.randn(4, 3, 8, 8))
        (diag + weak).backward()
        self.assertTrue(torch.isfinite(diag).item() and torch.isfinite(weak).item())
        self.assertLessEqual(float(logs["weight_max"]), 1.0 / args.weak_mixture_alpha + 1e-6)
        grads = [p.grad for p in model.net.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_multi_step_sampler_uses_one_call_per_step(self):
        torch.manual_seed(43)
        model = MeanFlow(SongUNet, settings("weak"), self.CONFIG)
        calls = []
        original = model.net_ema.forward

        def counting(z, tr, aug_cond=None):
            calls.append(float(tr[1][0]))
            return original(z, tr, aug_cond)

        model.net_ema.forward = counting
        noise = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            model.sample((2, 3, 8, 8), device="cpu", num_steps=4, initial_noise=noise)
        self.assertEqual(len(calls), 4)
        for value in calls:
            self.assertAlmostEqual(value, 0.25, places=5)

    def test_same_noise_gives_same_one_step_output(self):
        torch.manual_seed(44)
        model = MeanFlow(SongUNet, settings("weak"), self.CONFIG)
        noise = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            a = model.sample((2, 3, 8, 8), device="cpu", num_steps=1, initial_noise=noise)
            b = model.sample((2, 3, 8, 8), device="cpu", num_steps=1, initial_noise=noise)
        torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
