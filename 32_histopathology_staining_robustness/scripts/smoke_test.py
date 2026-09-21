#!/usr/bin/env python
"""Pre-flight smoke test / benchmark -- run this BEFORE committing 10 hours.

Answers, in order, the questions that decide whether a long Kaggle run is worth
starting:

1. **Is the environment sane?** CUDA present, AMP available, split CSVs present,
   domain firewall intact (checked structurally, without loading ``test_ood``).
2. **Is the data pipeline fast enough?** Measures pure DataLoader throughput for
   the configured normalisation + augmentation.  If the loader cannot feed the
   GPU, no schedule will save the run, and the fix is the offline
   normalisation cache (``scripts/preprocess_normalize.py``).
3. **Does one forward/backward pass fit and how long does it take?** Runs the
   real model, real loss and real AMP scaler for ``--steps`` optimizer steps.
4. **What does an epoch cost?** Extrapolates time-per-epoch and the number of
   epochs that fit inside the configured budget, then prints a verdict.

Nothing here writes into the checkpoint directories, so it is safe to run
immediately before the real job in the same session.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from histo_robust.data import build_dataset, describe_splits, resolve_reference  # noqa: E402
from histo_robust.models import BACKBONE_REGISTRY, build_classifier, estimate_backbone_vram_note  # noqa: E402
from histo_robust.utils.config import config_hash, load_config  # noqa: E402
from histo_robust.utils.kaggle import assert_gpu, configure_logging, log_environment  # noqa: E402
from histo_robust.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger("smoke_test")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=10, help="Optimizer steps to time.")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Steps excluded from the timing average.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-gpu-assert", action="store_true")
    parser.add_argument("--output", default=None, help="Where to write the JSON report.")
    parser.add_argument("--save-samples", type=int, default=0, help="Save N sample tiles per split for visual inspection.")
    parser.add_argument("--samples-dir", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args(argv)


def benchmark_loader(loader: Any, batches: int = 5) -> Dict[str, Any]:
    """Time pure batch production (no GPU work) to expose CPU bottlenecks."""
    started = time.time()
    fetched = 0
    images_seen = 0
    for batch in loader:
        fetched += 1
        images_seen += int(batch["image"].shape[0])
        if fetched >= batches:
            break
    elapsed = time.time() - started
    return {
        "batches_timed": fetched,
        "images_timed": images_seen,
        "seconds": round(elapsed, 4),
        "images_per_second": round(images_seen / max(1e-6, elapsed), 2),
        "seconds_per_batch": round(elapsed / max(1, fetched), 4),
    }


def save_sample_sheet(
    cfg: Dict[str, Any],
    reference: Any,
    out_dir: Path,
    class_names: Sequence[str],
    n: int = 4,
) -> List[str]:
    """Dump raw vs normalised tiles so the human can eyeball the pipeline."""
    from PIL import Image

    from histo_robust.utils.visualization import plot_sample_grid

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    raw_ds = build_dataset(cfg, "val_id", train=False, reference=None, repo_root=REPO_ROOT, limit=n)
    norm_ds = build_dataset(
        cfg, "val_id", train=False, reference=reference, repo_root=REPO_ROOT, limit=n
    )
    aug_ds = build_dataset(
        cfg, "val_id", train=True, reference=reference, repo_root=REPO_ROOT, limit=n
    )

    def to_uint8(tensor: Any) -> np.ndarray:
        arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        arr = arr * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    raw_tiles = [to_uint8(raw_ds[i]["image"]) for i in range(len(raw_ds))]
    norm_tiles = [to_uint8(norm_ds[i]["image"]) for i in range(len(norm_ds))]
    aug_tiles = [to_uint8(aug_ds[i]["image"]) for i in range(len(aug_ds))]

    for name, tiles in (("raw", raw_tiles), ("normalised", norm_tiles), ("augmented", aug_tiles)):
        path = out_dir / f"sample_{name}.png"
        plot_sample_grid(
            np.stack(tiles), path, titles=[f"{name} {i}" for i in range(len(tiles))],
            suptitle=f"{cfg.get('normalization', {}).get('name')} / {cfg.get('augmentation', {}).get('policy')}",
        )
        written.append(str(path))

    # Individual tiles are easier to inspect at full size in the Kaggle viewer.
    for idx, tile in enumerate(norm_tiles):
        path = out_dir / f"tile_{idx:02d}_normalised.png"
        Image.fromarray(tile).save(path)
        written.append(str(path))
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.set, repo_root=REPO_ROOT)
    if args.batch_size:
        cfg.setdefault("data", {})["batch_size"] = args.batch_size
    if args.weights_dir:
        cfg.setdefault("paths", {})["weights_dir"] = args.weights_dir
    if args.reference:
        cfg.setdefault("normalization", {})["reference_path"] = args.reference
    if args.search_root:
        existing = list((cfg.get("paths", {}) or {}).get("search_roots", []) or [])
        cfg.setdefault("paths", {})["search_roots"] = existing + list(args.search_root)

    out_path = Path(args.output) if args.output else Path("results/metrics/preflight_benchmark.json")
    configure_logging(out_path.parent / "preflight.log")
    set_seed(int((cfg.get("runtime", {}) or {}).get("seed", 42)))

    report: Dict[str, Any] = {
        "config_path": str(cfg.get("_config_path")),
        "config_hash": config_hash(cfg),
        "backbone": (cfg.get("model", {}) or {}).get("backbone"),
        "normalization": (cfg.get("normalization", {}) or {}).get("name"),
        "augmentation": (cfg.get("augmentation", {}) or {}).get("policy"),
        "batch_size": (cfg.get("data", {}) or {}).get("batch_size"),
        "grad_accum_steps": (cfg.get("train", {}) or {}).get("grad_accum_steps"),
        "num_workers": (cfg.get("data", {}) or {}).get("num_workers"),
    }

    environment = log_environment(logger)
    report["environment"] = environment
    if not args.skip_gpu_assert:
        assert_gpu()
    elif not environment.get("cuda_available"):
        logger.warning("Running the pre-flight benchmark on CPU -- timings will not extrapolate.")

    # --- structural split + firewall checks (no test_ood dataset is built) -----
    splits = describe_splits(cfg, repo_root=REPO_ROOT)
    report["splits"] = splits
    for required in ("train", "val_id", "test_id", "test_ood"):
        info = splits.get(required, {})
        if not info.get("exists"):
            logger.error("Split CSV missing: %s (%s). Run scripts/prepare_splits.py first.", required, info.get("path"))
            return 2
    for guarded in ("train", "val_id"):
        if splits[guarded].get("domain") != "source":
            logger.error("DOMAIN FIREWALL VIOLATION: %s has domain=%s", guarded, splits[guarded].get("domain"))
            return 3
    if splits["test_ood"].get("domain") != "target":
        logger.error("test_ood must be the target domain; found %s", splits["test_ood"].get("domain"))
        return 3
    logger.info("split + firewall checks: PASSED")

    reference, provenance = resolve_reference(cfg, repo_root=REPO_ROOT)
    report["reference_provenance"] = provenance

    backbone_key = str((cfg.get("model", {}) or {}).get("backbone", "resnet50")).lower()
    spec = BACKBONE_REGISTRY.get(backbone_key, {})
    if spec:
        logger.info(estimate_backbone_vram_note(spec, int((cfg.get("data", {}) or {}).get("batch_size", 64))))

    # --- 1) pure data pipeline throughput -------------------------------------
    import torch

    batch_size = int((cfg.get("data", {}) or {}).get("batch_size", 64))
    loader = torch.utils.data.DataLoader(
        build_dataset(cfg, "train", train=True, reference=reference, repo_root=REPO_ROOT, limit=batch_size * 5),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int((cfg.get("data", {}) or {}).get("num_workers", 4)),
        pin_memory=bool((cfg.get("data", {}) or {}).get("pin_memory", True)),
    )
    loader_started = time.time()
    data_stats = benchmark_loader(loader, batches=5)
    report["data_pipeline"] = data_stats
    report["data_pipeline"]["initialisation_seconds"] = round(time.time() - loader_started - data_stats["seconds"], 3)
    report["data_pipeline"]["normalizer_stats"] = loader.dataset.normalizer_stats()
    logger.info("data pipeline: %s", json.dumps(data_stats))

    # --- 2) a real forward/backward timing loop -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(
        cfg,
        pretrained=bool((cfg.get("model", {}) or {}).get("pretrained", True)),
        weights_dir=(cfg.get("paths", {}) or {}).get("weights_dir"),
    ).to(device)
    model.train(True)

    train_cfg = dict(cfg.get("train", {}) or {})
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0))
    )
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            lr=float(train_cfg.get("lr", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
            head_lr_mult=float(train_cfg.get("head_lr_mult", 1.0)),
        )
    )
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    step_times: List[float] = []
    loader_iter = iter(loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        started = time.time()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - started
        step_times.append(elapsed)
        logger.info("step %d/%d: %.3f s (loss=%.4f)", step + 1, args.steps, elapsed, float(loss.item()))

    measured = step_times[args.warmup_steps :] or step_times
    mean_step = float(np.mean(measured))
    report["train_step"] = {
        "steps": args.steps,
        "warmup_steps_excluded": args.warmup_steps,
        "step_seconds_mean": round(mean_step, 4),
        "step_seconds_min": round(float(np.min(measured)), 4),
        "step_seconds_max": round(float(np.max(measured)), 4),
        "images_per_second": round(batch_size / max(1e-6, mean_step), 2),
    }
    if device.type == "cuda":
        report["train_step"]["peak_gpu_memory_gb"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        )
        report["train_step"]["gpu_name"] = torch.cuda.get_device_name(0)

    # --- 3) extrapolation + verdict ------------------------------------------
    train_rows = int(splits["train"].get("n_samples", 0))
    accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    steps_per_epoch = max(1, int(np.ceil(train_rows / max(1, batch_size * accum))))
    epoch_seconds = steps_per_epoch * mean_step
    budget_minutes = float(train_cfg.get("max_train_minutes", 630))
    epochs_in_budget = budget_minutes * 60.0 / max(1e-6, epoch_seconds)

    data_seconds_per_batch = float(data_stats["seconds_per_batch"])
    gpu_bound = mean_step > data_seconds_per_batch * 0.9
    verdict = "OK"
    advice = "Pipeline looks healthy; safe to start the long run."
    if data_seconds_per_batch > mean_step:
        verdict = "CPU-BOUND"
        advice = (
            "DataLoader is slower than the GPU step. Pre-normalise the splits with "
            "scripts/preprocess_normalize.py, raise data.num_workers, or lower "
            "data.image_size before committing to a 10-hour run."
        )
    elif not gpu_bound:
        verdict = "IDLE-GPU"
        advice = (
            "The GPU waits on the loader between steps. Raise data.num_workers "
            "(4-8 on Kaggle) or enable the offline normalisation cache."
        )
    if epochs_in_budget < 3:
        advice += (
            f" Only {epochs_in_budget:.1f} epochs fit in {budget_minutes:.0f} min; "
            f"consider the 25k subset, freeze_backbone=true, or a larger batch."
        )

    report["extrapolation"] = {
        "train_rows": train_rows,
        "steps_per_epoch": steps_per_epoch,
        "seconds_per_epoch": round(epoch_seconds, 2),
        "minutes_per_epoch": round(epoch_seconds / 60.0, 3),
        "budget_minutes": budget_minutes,
        "epochs_fitting_in_budget": round(epochs_in_budget, 2),
        "planned_epochs": int(train_cfg.get("epochs", 12)),
        "full_matrix_estimate_minutes": round(
            epoch_seconds * int(train_cfg.get("epochs", 12)) * 13 / 60.0, 1
        ),
        "verdict": verdict,
        "advice": advice,
    }

    if args.save_samples:
        samples_dir = Path(args.samples_dir) if args.samples_dir else Path("results/samples")
        report["sample_files"] = save_sample_sheet(
            cfg,
            reference,
            samples_dir,
            list(cfg.get("data", {}).get("class_names", [])),
            n=int(args.save_samples),
        )
        logger.info("wrote sample tiles to %s", samples_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 78)
    print("PRE-FLIGHT BENCHMARK")
    print("=" * 78)
    print(f"  device              : {report['train_step'].get('gpu_name', 'CPU')}")
    print(f"  batch size          : {batch_size} (x{accum} accum = {batch_size * accum} effective)")
    print(f"  loader throughput   : {data_stats['images_per_second']} img/s ({data_stats['seconds_per_batch']} s/batch)")
    print(f"  train step          : {mean_step:.3f} s ({report['train_step']['images_per_second']} img/s)")
    if "peak_gpu_memory_gb" in report["train_step"]:
        print(f"  peak GPU memory     : {report['train_step']['peak_gpu_memory_gb']} GB")
    print(f"  minutes per epoch   : {report['extrapolation']['minutes_per_epoch']}")
    print(f"  epochs in budget    : {report['extrapolation']['epochs_fitting_in_budget']} of {report['extrapolation']['planned_epochs']} planned")
    print(f"  VERDICT             : {verdict}")
    print(f"  advice              : {advice}")
    print(f"  report              : {out_path}")
    print("=" * 78 + "\n")
    return 0 if verdict == "OK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
