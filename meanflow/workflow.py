"""Single-GPU step-based training and resumable multi-NFE FID.

Uses the repository's model, weak-form loss and EMA; no additional learning
objective lives here. Everything is driven from a JSON config so that a
notebook only has to call `run_experiment.py`.

Two properties this file is responsible for:

*   Resume safety. Training and evaluation both rerun from the last completed
    save, and refuse to continue a run whose config, code or data changed.
*   Peak-memory control. For `method=weak` the diagonal and weak terms are
    backwarded separately, so the largest live autograd graph is one term
    rather than both. Gradients are identical to summing first.
"""

import hashlib
import copy
import json
import math
import os
import platform
import subprocess
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.model_configs import instantiate_model

FORMAT = 3
FEATURE_DIM = 2048

# Keys that every run must pin, with their defaults.
CONFIG_DEFAULTS = {
    "method": "weak",
    "arch": "unet",
    "model_channels": 32,
    "batch_size": 256,
    "seed": 0,
    "lr": 3e-4,
    "warmup_steps": 1000,
    "ema_decay": 0.9999,
    "ema_decays": [],
    "snapshot_steps": [],
    "step_rng": False,
    "horizontal_flip": True,
    "grad_clip": 0.0,
    # weak-form objective
    "weak_features": 64,
    "weak_weight": 1.0,
    "diag_weight": 1.0,
    "weak_sigma_z": 1.0,
    "weak_sigma_r": 1.0,
    "weak_sigma_t": 1.0,
    "weak_fp64": False,
    "weak_test_family": "vanishing",
    "weak_time_sampler": "uniform",
    "weak_time_correction": "importance",
    "weak_mixture_alpha": 0.2,
    "weak_P_mean_t": -0.6,
    "weak_P_std_t": 1.6,
    "weak_P_mean_r": -4.0,
    "weak_P_std_r": 1.6,
    "weak_diag_time": "uniform",
    # strong-form baselines only
    "diag_probability": None,
}

WEAK_ONLY = {
    "weak_features", "weak_weight", "diag_weight", "weak_sigma_z", "weak_sigma_r",
    "weak_sigma_t", "weak_fp64", "weak_test_family", "weak_time_sampler",
    "weak_time_correction", "weak_mixture_alpha", "weak_P_mean_t", "weak_P_std_t",
    "weak_P_mean_r", "weak_P_std_r", "weak_diag_time",
}

# Used only by the existing original MeanFlow loss in models/meanflow.py.
# Keep these separate so existing weak checkpoint configs stay unchanged.
MF_DEFAULTS = {
    "tr_sampler": "v0",
    "ratio": 0.75,
    "P_mean_t": -2.0,
    "P_std_t": 2.0,
    "P_mean_r": -2.0,
    "P_std_r": 2.0,
    "norm_p": 0.75,
    "norm_eps": 1e-3,
}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def load_config(path):
    with open(path) as handle:
        return validate_config(json.load(handle))


def validate_config(config):
    unknown = set(config) - (set(CONFIG_DEFAULTS) | set(MF_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    merged = dict(CONFIG_DEFAULTS)
    merged.update(config)

    method = merged["method"]
    if method not in {"weak", "mf_control", "imf_diag", "mf"}:
        raise ValueError(f"Unsupported method: {method}")

    if method != "mf":
        mf_keys = sorted(set(config) & set(MF_DEFAULTS))
        if mf_keys:
            raise ValueError(f"mf-only keys set for method={method}: {mf_keys}")

    if method == "weak":
        # (b) diag_probability is a strong-form knob. Silently carrying it in a
        # weak config makes the run look configured when nothing reads it.
        if merged["diag_probability"] is not None:
            raise ValueError(
                "diag_probability has no effect when method=weak; the weak "
                "objective always evaluates its diagonal term. Remove the key "
                "and set diag_weight instead."
            )
        if merged["weak_test_family"] not in {"vanishing", "endpoint"}:
            raise ValueError("weak_test_family must be 'vanishing' or 'endpoint'")
        if merged["weak_time_sampler"] not in {"uniform", "logitnormal"}:
            raise ValueError("weak_time_sampler must be 'uniform' or 'logitnormal'")
        if merged["weak_time_correction"] not in {"importance", "adjoint"}:
            raise ValueError("weak_time_correction must be 'importance' or 'adjoint'")
        if not 0.0 < merged["weak_mixture_alpha"] <= 1.0:
            raise ValueError("weak_mixture_alpha must lie in (0, 1]")
        if (merged["weak_test_family"] == "endpoint"
                and merged["weak_time_sampler"] != "uniform"
                and merged["weak_time_correction"] == "adjoint"):
            raise ValueError(
                "endpoint test functions need weak_time_correction=importance; "
                "the adjoint form reweights the boundary term by q(r,1)."
            )
    else:
        leftovers = sorted(k for k in WEAK_ONLY if k in config)
        # A merged/saved config contains the ENTIRE default weak block even
        # for another method. Accept that block on re-validation, while still
        # rejecting explicitly supplied individual weak settings or overrides.
        default_weak_block = (
            WEAK_ONLY.issubset(config)
            and all(config[k] == CONFIG_DEFAULTS[k] for k in WEAK_ONLY)
        )
        if leftovers and not default_weak_block:
            raise ValueError(f"weak-only keys set for method={method}: {leftovers}")
        if method in {"mf_control", "imf_diag"} and merged["diag_probability"] is None:
            raise ValueError(f"method={method} requires diag_probability")

    if method == "mf":
        if merged["diag_probability"] is not None:
            raise ValueError("method=mf uses ratio, not diag_probability")
        for key, default in MF_DEFAULTS.items():
            merged.setdefault(key, default)
        if merged["tr_sampler"] not in {"v0", "v1"}:
            raise ValueError("tr_sampler must be v0 or v1")
        if not 0 <= merged["ratio"] <= 1:
            raise ValueError("ratio must be in [0, 1]")
        if not all(math.isfinite(merged[k]) for k in MF_DEFAULTS if k != "tr_sampler"):
            raise ValueError("MeanFlow numeric settings must be finite")
        if merged["P_std_t"] <= 0 or merged["P_std_r"] <= 0:
            raise ValueError("P_std_t and P_std_r must be positive")
        if merged["norm_p"] < 0 or merged["norm_eps"] <= 0:
            raise ValueError("Require norm_p >= 0 and norm_eps > 0")

    if merged["batch_size"] < 2:
        raise ValueError("The U statistic needs at least two samples per batch")
    if merged["model_channels"] < 1 or merged["lr"] <= 0:
        raise ValueError("model_channels and lr must be positive")
    if any(not 0 <= b < 1 for b in [merged["ema_decay"], *merged["ema_decays"]]):
        raise ValueError("All EMA decays must be in [0, 1)")
    if any(type(s) is not int or s < 1 for s in merged["snapshot_steps"]):
        raise ValueError("snapshot_steps must contain positive integers")
    if type(merged["step_rng"]) is not bool:
        raise ValueError("step_rng must be a boolean")
    if method == "weak" and (merged["weak_weight"] < 0 or merged["diag_weight"] <= 0
                              or merged["weak_features"] < 1):
        raise ValueError("Require weak_weight >= 0, diag_weight > 0, weak_features >= 1")
    return merged


def model_args(config):
    args = SimpleNamespace(
        arch=config["arch"],
        model_channels=config["model_channels"],
        dropout=0.0,
        use_edm_aug=False,
        method=config["method"],
        ema_decay=config["ema_decay"],
        ema_decays=list(config.get("ema_decays", [])),
        diag_probability=config["diag_probability"],
    )
    for key in WEAK_ONLY:
        setattr(args, key, config[key])
    if config["method"] == "mf":
        for key, default in MF_DEFAULTS.items():
            setattr(args, key, config.get(key, default))
    return args


def digest_json(payload):
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def source_identity():
    root = Path(__file__).resolve().parent
    files = sorted(p for p in root.rglob("*.py") if ".ipynb_checkpoints" not in str(p))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root.parent, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    return {"source_sha256": digest.hexdigest(), "commit": commit,
            "file_count": len(files)}


def environment_identity(device):
    name = torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else "cpu"
    versions = {"torch": torch.__version__, "python": platform.python_version()}
    for package in ("torchvision", "torchmetrics", "torch_fidelity", "numpy"):
        try:
            versions[package] = __import__(package).__version__
        except Exception:
            versions[package] = "missing"
    return {"gpu": name, "versions": versions}


# ---------------------------------------------------------------------------
# atomic IO
# ---------------------------------------------------------------------------


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_text(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"Checkpoint format {payload.get('format')} != {FORMAT}")
    return payload


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_cifar10(data_root):
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=str(data_root), train=True, download=True)
    array = np.asarray(dataset.data, dtype=np.uint8)          # [N, 32, 32, 3]
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def training_batch(pixels, step, config, device):
    """Sample with replacement from a per-step generator, optional h-flip."""
    generator = torch.Generator().manual_seed(
        (config["seed"] * 1_000_003 + step) % (2 ** 63 - 1))
    index = torch.randint(len(pixels), (config["batch_size"],), generator=generator)
    batch = pixels[index].to(device, non_blocking=True).float()
    batch = batch / 127.5 - 1.0
    if config["horizontal_flip"]:
        flip = torch.rand(len(batch), generator=generator) < 0.5
        batch[flip.to(device)] = batch[flip.to(device)].flip(-1)
    return batch


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def learning_rate(step, config):
    if step >= config["warmup_steps"]:
        return config["lr"]
    fraction = (step + 1) / max(config["warmup_steps"], 1)
    return 1e-8 + (config["lr"] - 1e-8) * fraction


def _backward_loss(model, x, config):
    """Return (total, logs) after populating .grad. Splits weak/diag."""
    if config["method"] == "weak":
        diag, weak, logs = model.loss_terms(x)
        for name, value in (("diag", diag), ("weak", weak)):
            if not torch.isfinite(value):
                raise FloatingPointError(f"Nonfinite {name} term")
        diag.backward()
        weak.backward()
        total = (diag + weak).detach()
    else:
        total = model.forward_with_loss(x, None)
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite loss")
        total.backward()
        logs = getattr(model, "last_losses", {})
        total = total.detach()
    return total, {k: float(v) for k, v in logs.items()}


def train(config, run_dir, data_root, target_steps, session_steps=None,
          max_minutes=240, save_every=500, save_minutes=10, log_every=100,
          device="cuda", pixels=None):
    """Rerun with an identical config to resume. target_steps may be raised."""
    config = validate_config(config)
    session_steps = session_steps or target_steps
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; choose a GPU runtime")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pixels = load_cifar10(data_root) if pixels is None else pixels
    data_hash = hashlib.sha256(pixels.contiguous().numpy().tobytes()).hexdigest()
    source = source_identity()
    identity = digest_json({"config": config, "source": source, "data": data_hash})

    checkpoint_path = run_dir / "checkpoint-last.pt"
    torch.manual_seed(config["seed"])
    model = instantiate_model(model_args(config)).to(device)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=config["lr"],
                                 betas=(0.9, 0.999))
    step, train_seconds = 0, 0.0

    if checkpoint_path.exists():
        saved = load_checkpoint(checkpoint_path)
        if saved["identity"] != identity:
            raise ValueError(
                "Run config, source or data changed. Use another run dir; "
                "this trainer does not silently resume a different experiment.")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step, train_seconds = saved["step"], saved["train_seconds"]
        if int(model.num_updates) != step:
            raise ValueError("EMA update count does not match the checkpoint step")
        if "rng_cpu" in saved:
            torch.set_rng_state(saved["rng_cpu"])
        if device.type == "cuda" and "rng_cuda" in saved:
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
        del saved
    elif (run_dir / "train.jsonl").exists():
        raise ValueError("A training log exists without a checkpoint; pick another run dir")

    atomic_text(json.dumps({"config": config, "source": source,
                            "environment": environment_identity(device),
                            "dataset_sha256": data_hash}, indent=2, sort_keys=True),
                run_dir / "environment" / "identity.json")

    log_path = run_dir / "train.jsonl"
    if log_path.exists():
        kept = []
        for line in log_path.read_text().splitlines():
            try:
                row = json.loads(line)
                if row["step"] <= step:
                    kept.append(json.dumps(row))
            except (ValueError, KeyError):
                pass  # drop a partially written line from an interrupted save
        atomic_text("\n".join(kept) + ("\n" if kept else ""), log_path)

    def save():
        if checkpoint_path.exists():
            os.replace(checkpoint_path, run_dir / "checkpoint-last.previous.pt")
        payload = {
            "format": FORMAT, "identity": identity, "config": config,
            "source": source, "dataset_sha256": data_hash,
            "image_shape": list(pixels.shape[1:]), "step": step,
            "train_seconds": train_seconds,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "rng_cpu": torch.get_rng_state(),
        }
        if device.type == "cuda":
            payload["rng_cuda"] = torch.cuda.get_rng_state_all()
        atomic_torch_save(payload, checkpoint_path)
        if step == 0 or step in config["snapshot_steps"]:
            snapshot = run_dir / "checkpoints" / f"step-{step:08d}.pt"
            if not snapshot.exists():
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                temporary = snapshot.with_suffix(".pt.tmp")
                shutil.copyfile(checkpoint_path, temporary)
                os.replace(temporary, snapshot)
        atomic_text(json.dumps({"step": step, "target_steps": target_steps,
                                "completed": step >= target_steps,
                                "train_seconds": train_seconds}, indent=2),
                    run_dir / "train_status.json")

    if step == 0:
        save()

    model.train()
    seconds_before = train_seconds
    started = last_save = time.time()
    stop_at = min(step + session_steps, target_steps)
    print(f"training {step} -> {stop_at} (target {target_steps})", flush=True)

    while step < stop_at:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, config)
        x = training_batch(pixels, step, config, device)
        if config["step_rng"]:
            # Loss draws are identical across paired weak_weight=0/1 runs and
            # do not depend on interruption/resume or diagnostic evaluation.
            torch.manual_seed((config["seed"] * 1_000_003 + step + 7_000_001) % (2 ** 63 - 1))
        optimizer.zero_grad(set_to_none=True)
        loss, logs = _backward_loss(model, x, config)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.net.parameters(),
            config["grad_clip"] if config["grad_clip"] > 0 else float("inf"),
            error_if_nonfinite=True)
        optimizer.step()
        model.update_ema()
        step += 1
        train_seconds = seconds_before + (time.time() - started)

        if step % log_every == 0 or step == stop_at:
            row = {"step": step, "loss": float(loss), "lr": learning_rate(step, config),
                   "grad_norm": float(grad_norm), **logs}
            with open(log_path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(row, flush=True)

        elapsed = time.time() - started
        due = (step % save_every == 0
               or step in config["snapshot_steps"]
               or time.time() - last_save > save_minutes * 60
               or step >= stop_at
               or elapsed > max_minutes * 60)
        if due:
            save()
            last_save = time.time()
        if elapsed > max_minutes * 60:
            print("session time limit reached", flush=True)
            break

    return {"step": step, "target_steps": target_steps,
            "completed": step >= target_steps}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def freeze_checkpoint(source_path, destination_dir):
    """Copy a checkpoint to a content-addressed file so evaluation is pinned."""
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(source_path)
    target = destination_dir / f"eval-{digest[:16]}.pt"
    if not target.exists():
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        atomic_torch_save(payload, target)
    return target


def load_for_eval(checkpoint_path, device):
    payload = load_checkpoint(checkpoint_path)
    config = validate_config(payload["config"])
    model = instantiate_model(model_args(config)).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


def select_weights(model, name="ema", payload=None, initial_checkpoint=None):
    """Select saved weights without copying raw weights over a saved EMA."""
    if name == "raw":
        return model.net
    if name == "ema":
        return model.net_ema
    if name.startswith("ema") and name[3:].isdigit() and int(name[3:]) >= 1:
        net = getattr(model, "net_" + name, None)
        if net is None:
            raise ValueError(f"Checkpoint has no {name}; inspect config.ema_decays")
        return net
    if name == "ema_noinit":
        if payload is None or not initial_checkpoint:
            raise ValueError("ema_noinit requires a saved --initial-checkpoint at step 0; no seed reconstruction is assumed")
        initial = load_checkpoint(initial_checkpoint)
        if initial["step"] != 0 or initial["identity"] != payload["identity"]:
            raise ValueError("Initial checkpoint must be step 0 of exactly the same run")
        updates = int(model.num_updates)
        if updates != payload["step"] or updates < 16:
            raise ValueError("EMA update count is inconsistent or no EMA update has occurred")
        coefficient = model.net_ema.ema_decay ** (16 * (updates // 16))
        result = copy.deepcopy(model.net_ema)
        with torch.no_grad():
            for key, parameter in result.named_parameters():
                origin = initial["model"]["net." + key].to(parameter.device).double()
                value = (parameter.double() - coefficient * origin) / (1 - coefficient)
                parameter.copy_(value.to(parameter.dtype))
        return result
    raise ValueError("weights must be raw, ema, ema1 (etc.), or ema_noinit")


def initial_noise(seed, batch_index, shape, device):
    """Same noise for every NFE, reproducible on resume."""
    generator = torch.Generator(device="cpu").manual_seed(
        (seed * 9_999_991 + batch_index) % (2 ** 63 - 1))
    return torch.randn(shape, generator=generator).to(device)


class FeatureStats:
    """float64 running sum / second moment of Inception features."""

    def __init__(self, dim=None):
        self.count = 0
        self.total = None if dim is None else torch.zeros(dim, dtype=torch.float64)
        self.outer = None if dim is None else torch.zeros(dim, dim, dtype=torch.float64)

    def _allocate(self, dim):
        # The width comes from the extractor, not from a constant, so a
        # stand-in feature extractor works without editing this class.
        if self.total is None:
            self.total = torch.zeros(dim, dtype=torch.float64)
            self.outer = torch.zeros(dim, dim, dtype=torch.float64)
        elif self.total.numel() != dim:
            raise ValueError(f"Feature width changed: {self.total.numel()} -> {dim}")

    def update(self, features):
        features = features.double().cpu()
        self._allocate(features.shape[1])
        self.count += features.shape[0]
        self.total += features.sum(dim=0)
        self.outer += features.T @ features

    def moments(self):
        if self.count < 2 or self.total is None:
            raise ValueError("Need at least two samples for a covariance")
        mean = self.total / self.count
        covariance = (self.outer - self.count * torch.outer(mean, mean)) / (self.count - 1)
        return mean, covariance

    def state(self):
        return {"count": self.count, "total": self.total, "outer": self.outer}

    @classmethod
    def from_state(cls, state):
        stats = cls(state["total"].numel())
        stats.count = int(state["count"])
        stats.total = state["total"].double()
        stats.outer = state["outer"].double()
        return stats


def _sqrtm_trace(first, second):
    """tr((S1 S2)^(1/2)) via a symmetric similarity; no scipy needed."""
    values, vectors = torch.linalg.eigh(first)
    root = vectors @ torch.diag(values.clamp_min(0).sqrt()) @ vectors.T
    middle = root @ second @ root
    middle = (middle + middle.T) / 2
    return torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum()


def frechet_distance(real, fake):
    mu_r, sigma_r = real.moments()
    mu_f, sigma_f = fake.moments()
    diff = (mu_r - mu_f).dot(mu_r - mu_f)
    return float(diff + torch.trace(sigma_r) + torch.trace(sigma_f)
                 - 2 * _sqrtm_trace(sigma_r, sigma_f))


def _inception(device):
    from torchmetrics.image.fid import NoTrainInceptionV3

    model = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"])
    return model.to(device).eval()


def _to_uint8(images):
    return ((images * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)


def evaluate_fid(checkpoint, output_dir, data_root, nfe_values, num_samples,
                 batch_size, seed, max_minutes=240, save_every_images=10_000,
                 save_minutes=10, device="cuda", pixels=None, weights="ema",
                 sampler="meanflow", num_real_samples=None, initial_checkpoint=None):
    """Resumable FID for several NFE values off one frozen checkpoint."""
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if num_samples < 2 or batch_size < 1:
        raise ValueError("Require num_samples >= 2 and batch_size >= 1")
    if not nfe_values or any(type(n) is not int or n < 1 for n in nfe_values) or len(set(nfe_values)) != len(nfe_values):
        raise ValueError("NFE values must be distinct positive integers")
    if sampler not in {"meanflow", "fm_euler"}:
        raise ValueError("Unknown sampler")

    model, payload = load_for_eval(checkpoint, device)
    net = select_weights(model, weights, payload, initial_checkpoint)
    pixels = load_cifar10(data_root) if pixels is None else pixels
    real_count = len(pixels) if num_real_samples is None else num_real_samples
    if not 2 <= real_count <= len(pixels):
        raise ValueError("num_real_samples must be between 2 and the reference dataset size")
    shape = (batch_size, *payload["image_shape"])
    protocol = {
        "format": 1, "checkpoint_sha256": file_sha256(checkpoint),
        "initial_sha256": file_sha256(initial_checkpoint) if initial_checkpoint else None,
        "weights": weights, "sampler": sampler, "nfe_values": sorted(nfe_values),
        "num_samples": num_samples, "num_real_samples": real_count,
        "batch_size": batch_size, "seed": seed,
        "real_sha256": hashlib.sha256(pixels[:real_count].contiguous().numpy().tobytes()).hexdigest(),
        "source": source_identity(), "environment": environment_identity(device),
        "features": "torch-fidelity Inception-2048 (tests may inject a stub)",
        "quantization": "round(255 * clamp(0.5*x+0.5,0,1))",
    }
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Evaluation protocol changed; use a new evaluation name/directory")
    elif any(output_dir.iterdir()):
        raise ValueError("Legacy or incomplete evaluation without protocol; use a new directory")
    else:
        atomic_text(json.dumps(protocol, indent=2, sort_keys=True), protocol_path)
    inception = _inception(device)
    results_path = output_dir / "fid_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    # --- reference statistics, computed once -------------------------------
    reference_path = output_dir / "stats-real.pt"
    if reference_path.exists():
        real = FeatureStats.from_state(torch.load(reference_path, weights_only=False))
    else:
        real = FeatureStats()
        with torch.no_grad():
            for start in range(0, real_count, batch_size):
                batch = pixels[start:min(start + batch_size, real_count)].to(device)
                real.update(inception(batch))
        atomic_torch_save(real.state(), reference_path)
    print(f"reference features: {real.count}", flush=True)
    if real.count != real_count:
        raise ValueError("Cached reference count does not match protocol")

    started = time.time()
    for nfe in nfe_values:
        key = str(nfe)
        if key in results:
            print(f"NFE {nfe}: already done, FID {results[key]['fid']:.3f}", flush=True)
            continue
        partial_path = output_dir / f"stats-fake-nfe{nfe}.pt"
        if partial_path.exists():
            partial = torch.load(partial_path, weights_only=False)
            fake = FeatureStats.from_state(partial)
            previous_seconds = float(partial.get("seconds", 0))
        else:
            fake = FeatureStats()
            previous_seconds = 0.0
        if fake.count > num_samples or (fake.count < num_samples and fake.count % batch_size):
            raise ValueError("Cached generated sample count is invalid")
        last_save, sample_clock = time.time(), time.time()
        def save_fake():
            state = fake.state()
            state["seconds"] = previous_seconds + time.time() - sample_clock
            atomic_torch_save(state, partial_path)
        while fake.count < num_samples:
            index = fake.count // batch_size
            current_shape = (min(batch_size, num_samples - fake.count), *shape[1:])
            noise = initial_noise(seed, index, current_shape, device)
            with torch.no_grad():
                images = model.sample(current_shape, net=net, device=device,
                                      num_steps=int(nfe), initial_noise=noise, sampler=sampler)
                if not torch.isfinite(images).all():
                    raise FloatingPointError("Generated images contain nonfinite values")
                uint8 = _to_uint8(images)
                fake.update(inception(uint8))
            if index == 0 and not (output_dir / f"preview-nfe{nfe}.png").exists():
                try:
                    from torchvision.utils import save_image
                    save_image((images[:64] * 0.5 + 0.5).clamp(0, 1),
                               output_dir / f"preview-nfe{nfe}.png", nrow=8)
                except Exception as error:   # a preview must never end a run
                    print(f"preview skipped: {error}", flush=True)
            due = (fake.count % save_every_images == 0
                   or time.time() - last_save > save_minutes * 60)
            if due:
                save_fake()
                last_save = time.time()
                rate = fake.count / max(time.time() - sample_clock, 1e-6)
                print(f"NFE {nfe}: {fake.count}/{num_samples} ({rate:.0f} img/s)", flush=True)
            if time.time() - started > max_minutes * 60:
                save_fake()
                print("session time limit reached; rerun to continue", flush=True)
                return {"completed": False, "results": results}
        save_fake()
        value = frechet_distance(real, fake)
        results[key] = {"fid": value, "n_fake": fake.count, "n_real": real.count,
                        "checkpoint_step": payload["step"],
                        "weights": weights, "sampler": sampler,
                        "seconds": previous_seconds + time.time() - sample_clock}
        atomic_text(json.dumps(results, indent=2, sort_keys=True), results_path)
        print(f"NFE {nfe}: FID {value:.3f}", flush=True)

    return {"completed": len(results) >= len(nfe_values), "results": results}


def sampling_cost(nfe_values, num_samples, num_real_samples=None):
    """The arithmetic that decides how long an evaluation takes."""
    per_nfe = {int(n): int(n) * num_samples for n in nfe_values}
    total = sum(per_nfe.values())
    return {"image_nfe_total": total, "per_nfe": per_nfe,
            "inception_images": num_samples * len(nfe_values) + (num_samples if num_real_samples is None else num_real_samples),
            "share": {k: v / total for k, v in per_nfe.items()}}
