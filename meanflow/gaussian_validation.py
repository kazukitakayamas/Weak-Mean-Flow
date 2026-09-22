"""Analytic Gaussian learning experiment; no CIFAR data, FID or pretrained teacher."""
import json
import math
from pathlib import Path

import torch
from torch import nn

from diagnostics import scalar_stats, write_csv
from models.weak_loss import weak_terms, sample_uniform_triangle
from workflow import CONFIG_DEFAULTS, model_args, atomic_text, source_identity


def gaussian_mean_velocity(z, r, t, sigma=0.5):
    """Exact mean velocity, including the continuous limit r=t.

    s_t^2=(1-t)^2 sigma^2+t^2; u=(1-s_r/s_t)z/(t-r).
    Rationalising avoids catastrophic cancellation near the diagonal.
    """
    st = ((1-t).square() * sigma**2 + t.square()).sqrt()
    sr = ((1-r).square() * sigma**2 + r.square()).sqrt()
    factor = ((1+sigma**2)*(t+r)-2*sigma**2) / (st*(st+sr))
    return factor * z


class ExactGaussian(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, z, times, aug_cond=None):
        t, h = (v.view(-1, 1, 1, 1) for v in times)
        return gaussian_mean_velocity(z, t-h, t, self.sigma)


class GaussianMLP(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim+2, 64), nn.SiLU(),
                                    nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, dim))

    def forward(self, z, times, aug_cond=None):
        t, h = times
        inputs = torch.cat([z.flatten(1), t[:, None], h[:, None]], dim=1)
        return self.layers(inputs).reshape_as(z)


@torch.no_grad()
def gaussian_metrics(net, sigma, dim, seed, device, count=4096):
    devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        x = sigma * torch.randn(count, dim, 1, 1, device=device)
        noise = torch.randn_like(x)
        t, r = sample_uniform_triangle(count, device)
        z = (1-t)*x+t*noise
        pred = net(z, (t.flatten(), (t-r).flatten()))
        exact = gaussian_mean_velocity(z, r, t, sigma)
        diagonal = net(z, (t.flatten(), torch.zeros_like(t).flatten()))
        v = gaussian_mean_velocity(z, t, t, sigma)
        one = torch.ones(count, device=device)
        generated = noise - net(noise, (one, one))
        # For this Gaussian the true one-step map is exactly sigma * noise.
        truth = sigma * noise
        return dict(mean_velocity_mse=float((pred-exact).square().mean()),
                    diagonal_velocity_mse=float((diagonal-v).square().mean()),
                    one_step_map_mse=float((generated-truth).square().mean()),
                    generated_mean=float(generated.mean()),
                    generated_variance=float(generated.var(unbiased=False)),
                    target_variance=sigma**2)


def run_gaussian(output_dir, steps=2000, batch=256, features=32, weights=(0., 1.),
                 sigma=0.5, dim=1, lr=1e-3, seed=0, device='cpu', oracle_repeats=64):
    if steps < 1 or batch < 2 or features < 1 or sigma <= 0 or dim < 1 or oracle_repeats < 2:
        raise ValueError('Invalid Gaussian experiment size')
    if not weights or any(w < 0 for w in weights) or len(set(weights)) != len(weights):
        raise ValueError('weak weights must be distinct and nonnegative')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(CONFIG_DEFAULTS, batch_size=batch, weak_features=features,
                  weak_test_family='endpoint', weak_time_sampler='logitnormal',
                  weak_time_correction='importance', weak_weight=1.0, weak_fp64=False)
    protocol = dict(steps=steps, batch=batch, features=features, weak_weights=list(weights),
                    sigma=sigma, dim=dim, lr=lr, seed=seed, device=str(device),
                    oracle_repeats=oracle_repeats, source=source_identity(),
                    weak_settings=config,
                    scope='Gaussian diagnostic, not CIFAR or FID; raw model, no EMA')
    path = output_dir / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Gaussian settings changed; use a new output directory')
    atomic_text(json.dumps(protocol, indent=2), path)
    args = model_args(config)
    oracle = ExactGaussian(sigma).to(device)
    values = []
    with torch.no_grad():
        for rep in range(oracle_repeats):
            torch.manual_seed(seed + 700_000 + rep)
            x = sigma * torch.randn(batch, dim, 1, 1, device=device)
            _, weak, _ = weak_terms(oracle, x, args)
            values.append(float(weak))
    oracle_result = dict(weak_estimator=scalar_stats(values),
                         **gaussian_metrics(oracle, sigma, dim, seed+500_000, device))
    traces, results = [], []
    for weight in weights:
        torch.manual_seed(seed)
        net = GaussianMLP(dim).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        args.weak_weight = float(weight)
        baseline = gaussian_metrics(net, sigma, dim, seed+500_000, device)
        traces.append(dict(weak_weight=weight, step=0, **baseline))
        for step in range(steps):
            torch.manual_seed(seed + 100_000 + step)
            x = sigma * torch.randn(batch, dim, 1, 1, device=device)
            diag, weak, _ = weak_terms(net, x, args)
            loss = diag+weak
            if not torch.isfinite(loss):
                raise FloatingPointError('Gaussian experiment diverged')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (step+1) % max(1, steps//10) == 0 or step+1 == steps:
                metrics = gaussian_metrics(net, sigma, dim, seed+500_000, device)
                traces.append(dict(weak_weight=weight, step=step+1, **metrics))
                print(f'Gaussian w={weight} step={step+1}: {metrics}', flush=True)
                write_csv(traces, output_dir / 'learning_curves.csv')
        results.append(dict(weak_weight=weight, initial=baseline,
                            final=gaussian_metrics(net, sigma, dim, seed+500_000, device)))
    result = dict(oracle=oracle_result, runs=results,
                  note='A small finite-run experiment is evidence about these settings, not a convergence proof.')
    atomic_text(json.dumps(result, indent=2, allow_nan=False), output_dir / 'results.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for weight in weights:
        group = [r for r in traces if r['weak_weight'] == weight]
        for ax, key in zip(axes, ('mean_velocity_mse', 'one_step_map_mse')):
            ax.plot([r['step'] for r in group], [r[key] for r in group], label=f'w={weight}')
            ax.set(xlabel='updates', ylabel=key, yscale='log')
            ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'learning_curves.png', dpi=150)
    plt.close(fig)
    return result
