"""Command-line entry point for the weak-form CIFAR-10 experiment.

    python meanflow/run_experiment.py preflight --config ... --run-dir ...
    python meanflow/run_experiment.py train     --config ... --run-dir ...
    python meanflow/run_experiment.py evaluate  --run-dir ... --evaluation-name fid50k
    python meanflow/run_experiment.py report    --run-dir ... --evaluation-name fid50k

The original epoch-based `train.py` is left untouched and is not used here.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow import (  # noqa: E402
    evaluate_fid, freeze_checkpoint, instantiate_model, load_cifar10,
    load_checkpoint, load_config, model_args, sampling_cost, train,
    _backward_loss, atomic_text, environment_identity,
)

DEFAULT_STEPS = [1, 2, 4, 8, 16, 32, 64, 128]


def _device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Select a GPU runtime.")
    return torch.device(name)


# ---------------------------------------------------------------------------


def cmd_preflight(args):
    device = _device(args.device)
    config = load_config(args.config)
    print(json.dumps({"config": config,
                      "environment": environment_identity(device)},
                     indent=2, sort_keys=True))

    pixels = load_cifar10(args.data_root)
    torch.manual_seed(config["seed"])
    probe = instantiate_model(model_args(config)).to(device).train()
    parameters = sum(p.numel() for p in probe.net.parameters())
    optimizer = torch.optim.Adam(probe.net.parameters(), lr=config["lr"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    from workflow import training_batch
    timings = []
    try:
        for step in range(3):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            x = training_batch(pixels, step, config, device)
            optimizer.zero_grad(set_to_none=True)
            loss, logs = _backward_loss(probe, x, config)
            optimizer.step()
            probe.update_ema()
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            print(f"  step {step}: loss {float(loss):+.5f} {logs}", flush=True)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            "Out of memory during preflight. Lower batch_size (or "
            "model_channels) in the config and start a NEW run dir.")

    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
    per_step = sum(timings[1:]) / max(len(timings) - 1, 1)
    print(json.dumps({
        "parameters_millions": round(parameters / 1e6, 3),
        "peak_vram_gib": round(peak, 2),
        "seconds_per_step": round(per_step, 4),
        "hours_for_target_steps": round(per_step * args.target_steps / 3600, 2),
    }, indent=2))
    print("\nThis is a short measurement. Drive I/O, initialisation and a "
          "shared GPU are not included.")

    del probe, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------


def cmd_train(args):
    device = _device(args.device)
    config = load_config(args.config)
    result = train(
        config, run_dir=args.run_dir, data_root=args.data_root,
        target_steps=args.target_steps, session_steps=args.session_steps,
        max_minutes=args.max_minutes, save_every=args.save_every,
        save_minutes=args.save_minutes, log_every=args.log_every, device=device,
    )
    print(json.dumps(result, indent=2))
    if not result["completed"]:
        print("Not finished. Rerun the same command to continue.")


# ---------------------------------------------------------------------------


def cmd_evaluate(args):
    device = _device(args.device)
    run_dir = Path(args.run_dir)
    evaluation_dir = run_dir / "evaluations" / args.evaluation_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    selection_path = evaluation_dir / "selection.json"

    settings = {"steps": sorted(args.steps), "num_samples": args.num_samples,
                "batch_size": args.batch_size, "seed": args.seed}

    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if args.checkpoint:
            raise SystemExit(
                f"'{args.evaluation_name}' is already pinned to "
                f"{selection['checkpoint']}. Use a new --evaluation-name for a "
                "different checkpoint.")
        if selection["settings"] != settings:
            raise SystemExit(
                f"'{args.evaluation_name}' was created with {selection['settings']}. "
                "Changing the FID settings needs a new --evaluation-name.")
        frozen = Path(selection["checkpoint"])
    else:
        source = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoint-last.pt"
        if not source.exists():
            raise SystemExit(f"No checkpoint at {source}. Train first.")
        payload = load_checkpoint(source)
        if payload["step"] < args.min_train_steps:
            raise SystemExit(
                f"Checkpoint is at {payload['step']} steps, below "
                f"--min-train-steps {args.min_train_steps}. Continue training, "
                "or lower the threshold on purpose.")
        frozen = freeze_checkpoint(source, run_dir / "eval-checkpoints")
        selection = {"checkpoint": str(frozen), "train_step": payload["step"],
                     "settings": settings}
        atomic_text(json.dumps(selection, indent=2, sort_keys=True), selection_path)
        del payload

    cost = sampling_cost(settings["steps"], args.num_samples)
    print(f"pinned checkpoint: {frozen} (train step {selection['train_step']})")
    print(f"generator cost: {cost['image_nfe_total']:,} image-NFE; "
          f"Inception on {cost['inception_images']:,} images")
    heavy = sorted(cost["share"].items(), key=lambda kv: -kv[1])[:2]
    print("  dominated by NFE " + ", ".join(f"{k} ({v:.0%})" for k, v in heavy))

    result = evaluate_fid(
        checkpoint=frozen, output_dir=evaluation_dir / "results",
        data_root=args.data_root, nfe_values=settings["steps"],
        num_samples=args.num_samples, batch_size=args.batch_size,
        seed=args.seed, max_minutes=args.max_minutes,
        save_every_images=args.save_every_images, device=device,
    )
    print(json.dumps({"completed": result["completed"]}, indent=2))
    if not result["completed"]:
        print("Not finished. Rerun the same command to continue.")


# ---------------------------------------------------------------------------


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluation_dir = Path(args.run_dir) / "evaluations" / args.evaluation_name
    selection = json.loads((evaluation_dir / "selection.json").read_text())
    results_path = evaluation_dir / "results" / "fid_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    rows = []
    for nfe in selection["settings"]["steps"]:
        entry = results.get(str(nfe), {})
        rows.append({"nfe": nfe, "fid": entry.get("fid"),
                     "n_fake": entry.get("n_fake"),
                     "n_real": entry.get("n_real"),
                     "checkpoint_step": entry.get("checkpoint_step"),
                     "seconds": entry.get("seconds")})

    header = "nfe,fid,n_fake,n_real,checkpoint_step,seconds"
    lines = [header] + [
        ",".join("" if row[k] is None else str(row[k]) for k in
                 ("nfe", "fid", "n_fake", "n_real", "checkpoint_step", "seconds"))
        for row in rows]
    csv_path = evaluation_dir / "fid_comparison.csv"
    atomic_text("\n".join(lines) + "\n", csv_path)

    print(f"{'NFE':>5} {'FID':>10} {'n_fake':>8}")
    for row in rows:
        value = "-" if row["fid"] is None else f"{row['fid']:.3f}"
        print(f"{row['nfe']:>5} {value:>10} {row['n_fake'] or '-':>8}")

    done = [(r["nfe"], r["fid"]) for r in rows if r["fid"] is not None]
    if done:
        figure, axis = plt.subplots(figsize=(7, 4.2))
        axis.plot([n for n, _ in done], [f for _, f in done], marker="o")
        axis.set_xscale("log", base=2)
        axis.set_xticks([n for n, _ in done])
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_xlabel("sampling steps (= NFE)")
        axis.set_ylabel("FID")
        axis.set_title(f"{args.evaluation_name}: train step {selection['train_step']}")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(evaluation_dir / "fid_vs_steps.png", dpi=150)
        plt.close(figure)
    print(f"\nCSV: {csv_path}")


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, config=True):
        if config:
            p.add_argument("--config", required=True)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--data-root", required=True)

    p = sub.add_parser("preflight")
    common(p)
    p.add_argument("--target-steps", type=int, default=20_000)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("train")
    common(p)
    p.add_argument("--target-steps", type=int, default=20_000)
    p.add_argument("--session-steps", type=int, default=20_000)
    p.add_argument("--max-minutes", type=float, default=240)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--save-minutes", type=float, default=10)
    p.add_argument("--log-every", type=int, default=100)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate")
    common(p, config=False)
    p.add_argument("--evaluation-name", required=True)
    p.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    p.add_argument("--num-samples", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--min-train-steps", type=int, default=20_000)
    p.add_argument("--max-minutes", type=float, default=240)
    p.add_argument("--save-every-images", type=int, default=10_000)
    p.add_argument("--checkpoint", default="")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("report")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--evaluation-name", required=True)
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
