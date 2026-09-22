"""Read-only checkpoint diagnostics. No optimizer update or checkpoint write."""
import copy
import csv
import io
import json
import math
from pathlib import Path

import torch

from models.weak_loss import weak_terms
from workflow import (atomic_text, file_sha256, source_identity, environment_identity,
                      load_cifar10, load_for_eval, model_args, select_weights,
                      initial_noise)


def write_csv(rows, path):
    if not rows:
        return
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(stream.getvalue(), path)


def scalar_stats(values):
    x = torch.tensor(values, dtype=torch.float64)
    sd = float(x.std()) if len(x) > 1 else None
    se = sd / math.sqrt(len(x)) if sd is not None else None
    return dict(n=len(x), mean=float(x.mean()), std=sd, mean_se=se,
                ci95_normal_approx=[float(x.mean()) - 1.96 * se,
                                    float(x.mean()) + 1.96 * se] if se is not None else None)


class GradientMoments:
    """Online vector mean and scalar total sample variance, on the CPU."""
    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, 0.0

    def update(self, gradient):
        x = gradient.detach().to(device="cpu", dtype=torch.float64)
        self.n += 1
        if self.mean is None:
            self.mean = x.clone()
        else:
            delta = x - self.mean
            self.mean += delta / self.n
            self.m2 += float(torch.dot(delta, x - self.mean))

    def summary(self):
        mean_sq = float(torch.dot(self.mean, self.mean))
        variance = max(0.0, self.m2 / (self.n - 1)) if self.n > 1 else None
        # ||sample mean||^2 is upward biased by variance / n. Preserve the
        # SIGN of the corrected estimate; nonpositive does not prove zero.
        signal = mean_sq - variance / self.n if variance is not None else None
        return dict(n=self.n, mean_gradient_norm=math.sqrt(mean_sq),
                    noise_rms=math.sqrt(variance) if variance is not None else None,
                    mean_noise_rms=math.sqrt(variance / self.n) if variance is not None else None,
                    signal_squared_unbiased=signal,
                    snr_estimate=math.sqrt(signal / variance)
                    if signal is not None and signal > 0 and variance > 0 else None,
                    note="Nonpositive signal estimate means unresolved at this repeat count, not zero gradient.")


def draw_features(dim, count, config, seed, device):
    # Always draw FP32 on CPU, then cast in the loss. This permits a paired
    # precision check without changing the samples or random coefficients.
    g = torch.Generator().manual_seed(seed)
    return (
        (torch.randn(count, dim, generator=g) * (config['weak_sigma_z'] / math.sqrt(dim))).to(device),
        (torch.randn(count, generator=g) * config['weak_sigma_r']).to(device),
        (torch.randn(count, generator=g) * config['weak_sigma_t']).to(device),
        (torch.rand(count, generator=g) * (2 * math.pi)).to(device),
    )


def draw_data(pixels, batch, seed, config, device):
    g = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(pixels), (batch,), generator=g)
    x = pixels[indices].to(device).float() / 127.5 - 1
    if config.get('horizontal_flip', False):
        mask = (torch.rand(batch, generator=g) < .5).to(device)
        x = torch.where(mask[:, None, None, None], x.flip(-1), x)
    return x


def loss_gradients(net, x, args, features, loss_seed):
    devices = [x.device.index or 0] if x.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(loss_seed)
        diag, weak, logs = weak_terms(net, x, args, feature_parameters=features)
        params = [p for p in net.parameters() if p.requires_grad]
        gd = torch.autograd.grad(diag, params, allow_unused=True)
        gw = torch.autograd.grad(weak, params, allow_unused=True)
        def flat(gs):
            return torch.cat([(v.detach() if v is not None else torch.zeros_like(p)).reshape(-1)
                              for v, p in zip(gs, params)]).cpu()
        return float(diag.detach()), float(weak.detach()), flat(gd), flat(gw), logs


def gradient_diagnostics(model, payload, pixels, output, weights, batch_sizes,
                         feature_counts, repeats=32, resampling=('both',), seed=31415,
                         precision_repeats=2, device='cuda'):
    if repeats < 2 or min(batch_sizes) < 2 or min(feature_counts) < 1:
        raise ValueError('repeats >= 2, batch sizes >= 2 and feature counts >= 1 required')
    if any(m not in {'both', 'data', 'features'} for m in resampling):
        raise ValueError('resampling must be both, data or features')
    net = select_weights(model, weights).eval()
    net.requires_grad_(True)
    config = payload['config']
    if config['method'] != 'weak':
        raise ValueError('Weak gradient diagnostics require a weak-method checkpoint')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, precision = [], []
    dim = pixels[0].numel()
    weight = config['weak_weight']
    diag_weight = config.get('diag_weight', 1.0)
    for batch in batch_sizes:
        for count in feature_counts:
            for mode in resampling:
                args = model_args(config)
                # Return unweighted terms even for the weak_weight=0 control.
                args.weak_weight, args.diag_weight = 1.0, 1.0
                args.weak_features = count
                args.weak_fp64 = bool(config['weak_fp64'])
                dm, wm, rows = GradientMoments(), GradientMoments(), []
                for rep in range(repeats):
                    data_id = 0 if mode == 'features' else rep
                    feature_id = 0 if mode == 'data' else rep
                    x = draw_data(pixels, batch, seed + 10_000 + data_id, config, device)
                    features = draw_features(dim, count, config, seed + 20_000 + feature_id, device)
                    loss_seed = seed + 30_000 + data_id
                    diag, weak, gd, gw, logs = loss_gradients(net, x, args, features, loss_seed)
                    gd64, gw64 = gd.double(), gw.double()
                    nd, nw = float(gd64.norm()), float(gw64.norm())
                    dm.update(gd)
                    wm.update(gw)
                    rows.append(dict(repeat=rep, diag_mse=diag, weak_u=weak,
                                     diag_grad_norm=nd, weak_grad_norm=nw,
                                     weighted_grad_ratio=weight * nw / max(diag_weight * nd, 1e-30),
                                     cosine=float(torch.dot(gd64, gw64)) / (nd * nw) if nd * nw > 0 else None,
                                     weight_max=float(logs['weight_max']), q_min=float(logs['q_min'])))
                    if rep < precision_repeats:
                        paired = copy.copy(args)
                        paired.weak_fp64 = not args.weak_fp64
                        _, other, _, other_grad, _ = loss_gradients(net, x, paired, features, loss_seed)
                        loss32, loss64 = (other, weak) if args.weak_fp64 else (weak, other)
                        g32, g64 = (other_grad.double(), gw64) if args.weak_fp64 else (gw64, other_grad.double())
                        precision.append(dict(batch=batch, features=count, mode=mode, repeat=rep,
                                              weak_fp32=loss32, weak_fp64=loss64,
                                              loss_absolute_difference=abs(loss32-loss64),
                                              weak_grad_relative_difference=float((g32-g64).norm()) / max(float(g64.norm()), 1e-30)))
                    del gd, gw, gd64, gw64, x, features
                    print(f'gradients {weights} B={batch} M={count} {mode}: {rep+1}/{repeats}', flush=True)
                stem = f'gradients_{weights}_b{batch}_m{count}_{mode}'
                write_csv(rows, output / f'{stem}.csv')
                record = dict(weights=weights, batch=batch, features=count, resampling=mode,
                              weak_weight=weight, diag_weight=diag_weight,
                              weak_loss=scalar_stats([r['weak_u'] for r in rows]),
                              diag_gradient=dm.summary(), weak_gradient=wm.summary(),
                              weighted_gradient_ratio=scalar_stats([r['weighted_grad_ratio'] for r in rows]),
                              cosine=scalar_stats([r['cosine'] for r in rows if r['cosine'] is not None])
                              if any(r['cosine'] is not None for r in rows) else None,
                              interpretation='data/features modes are conditional sensitivity probes, not an additive variance decomposition; gradients are before Adam preconditioning')
                summaries.append(record)
                atomic_text(json.dumps(summaries, indent=2, allow_nan=False), output / 'gradient_summary.json')
                write_csv(precision, output / 'precision_comparison.csv')
    return summaries


@torch.no_grad()
def interval_diagnostics(model, payload, pixels, output, weights=('raw', 'ema'),
                         batch=64, seed=2718, device='cuda'):
    if batch < 2:
        raise ValueError('interval batch must be >= 2')
    from torchvision.utils import save_image
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = payload['config']
    x = draw_data(pixels, batch, seed, config, device)
    noise = initial_noise(seed, 0, tuple(x.shape), device)
    # Average raw reference pixels without a second full float dataset copy.
    pixel_sum = torch.zeros_like(pixels[0], dtype=torch.float64)
    for chunk in pixels.split(1000):
        pixel_sum += chunk.double().sum(0)
    mean_image = (pixel_sum / len(pixels) / 127.5 - 1).to(device).float()
    intervals, previews, ema_info = [], [], []
    for name in weights:
        net = select_weights(model, name).eval()
        for time in (0.25, 0.5, 0.75, 1.0):
            t = torch.full((batch,), time, device=device)
            z = x * (1-time) + noise * time
            base = net(z, (t, torch.zeros_like(t)), None)
            for fraction in (0., .25, .5, .75, 1.):
                h = t * fraction
                pred = net(z, (t, h), None)
                diff = float((pred-base).square().mean())
                intervals.append(dict(weights=name, t=time, h=float(time*fraction),
                                      h_fraction=fraction, difference_from_diagonal_mse=diff,
                                      relative_difference=diff / max(float(base.square().mean()), 1e-30),
                                      diagonal_cfm_mse=float((base-(noise-x)).square().mean())))
        for sampler in ('meanflow', 'fm_euler'):
            for nfe in (1, 128):
                img = model.sample(tuple(noise.shape), net=net, device=device,
                                   num_steps=nfe, initial_noise=noise, sampler=sampler)
                if not torch.isfinite(img).all():
                    raise FloatingPointError('Nonfinite interval preview')
                path = output / f'{name}_{sampler}_nfe{nfe}.png'
                save_image((img * .5 + .5).clamp(0, 1), path, nrow=8)
                previews.append(dict(weights=name, sampler=sampler, nfe=nfe,
                                     pixel_std=float(img.std()),
                                     mse_to_data_mean=float((img-mean_image).square().mean()),
                                     between_sample_variance=float(img.var(dim=0).mean()),
                                     clipped_fraction=float(((img < -1) | (img > 1)).float().mean())))
        if name != 'raw':
            beta = float(net.ema_decay)
            count = int(model.num_updates)
            ema_info.append(dict(weights=name, beta=beta, optimizer_updates=count,
                                 ema_updates=count//16,
                                 initial_parameter_coefficient=beta**(16*(count//16))))
    write_csv(intervals, output / 'interval_dependence.csv')
    write_csv(previews, output / 'preview_metrics.csv')
    atomic_text(json.dumps(ema_info, indent=2), output / 'ema_information.json')
    return dict(intervals=intervals, previews=previews)


def diagnose(checkpoint, output, data_root, mode='all', weights='raw', repeats=32,
             batch_sizes=None, feature_counts=None, resampling=('both',), seed=31415,
             interval_batch=64, device='cuda', pixels=None, preview_weights=('raw', 'ema')):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, payload = load_for_eval(checkpoint, device)
    pixels = load_cifar10(data_root) if pixels is None else pixels
    batch_sizes = batch_sizes or [payload['config']['batch_size']]
    feature_counts = feature_counts or [payload['config']['weak_features']]
    protocol = dict(checkpoint_sha256=file_sha256(checkpoint), checkpoint_step=payload['step'],
                    mode=mode, weights=weights, repeats=repeats, batch_sizes=batch_sizes,
                    feature_counts=feature_counts, resampling=list(resampling), seed=seed,
                    interval_batch=interval_batch, preview_weights=list(preview_weights),
                    pixel_sha256=__import__('hashlib').sha256(pixels.numpy().tobytes()).hexdigest(),
                    source=source_identity(), environment=environment_identity(device))
    target = output / 'protocol.json'
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError('Diagnostic settings changed; use a new --output-dir')
    atomic_text(json.dumps(protocol, indent=2), target)
    if mode in ('all', 'interval'):
        interval_diagnostics(model, payload, pixels, output, weights=preview_weights,
                             batch=interval_batch, seed=seed, device=device)
    if mode in ('all', 'gradients'):
        gradient_diagnostics(model, payload, pixels, output, weights=weights,
                             batch_sizes=batch_sizes, feature_counts=feature_counts,
                             repeats=repeats, resampling=resampling, seed=seed, device=device)
    return dict(output_dir=str(output), checkpoint_step=payload['step'], completed=True)
