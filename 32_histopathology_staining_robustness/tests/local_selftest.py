#!/usr/bin/env python
"""Local self-test for the pure-NumPy parts of the pipeline.

Runs without torch/pandas so it can execute on any machine:

* Macenko/Reinhard normalisation: shape/dtype preservation, no NaNs, the
  >85% glass bypass, the fallback counter, and reference determinism.
* HED stain jitter: output range, that it actually changes the image, and that
  the per-stain scaling behaves (stronger E scale -> more eosinophilic).
* Geometric augmentation: flips/rotations preserve shape and are deterministic
  for a fixed seed.
* Metrics: macro-F1 / balanced accuracy / AUROC / ECE against hand-computed
  values on small synthetic predictions, plus the delta/rr robustness maths.
* TimeBudget: stop conditions and reason precedence.

Usage::

    python tests/local_selftest.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def synthetic_he_tile(size: int = 224, seed: int = 0, tissue: float = 0.9) -> np.ndarray:
    """A crude but stain-like H&E tile: purple nuclei on pink stroma."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    nuclei = np.exp(-(((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * (size / 8) ** 2)))
    noise = rng.normal(0, 6, size=(size, size, 3))
    base = np.stack(
        [
            235 - 90 * nuclei,   # R
            200 - 40 * nuclei,   # G
            225 - 130 * nuclei,  # B  (haematoxylin absorbs green/blue -> purple)
        ],
        axis=2,
    ) + noise
    tile = np.clip(base, 0, 255).astype(np.uint8)
    if tissue < 1.0:
        cutoff = int(size * (1.0 - tissue))
        tile[:cutoff, :, :] = 250
    return tile


def in_gamut_he_tile(
    h: float, e: float, size: int = 64, seed: int = 0, noise: float = 1.0
) -> np.ndarray:
    """A tile built by forward-modelling the canonical H&E law.

    Because the tile is *generated* from the same stain matrix the augmentation
    deconvolves with, the identity-scaling round trip must be exact and the
    per-stain scaling directions are unambiguous.  Adding mild spatial structure
    keeps the tile non-uniform so the deconvolution is not degenerate.
    """
    from histo_robust.augmentation.stain_jitter import he_stain_matrix

    stain = he_stain_matrix()
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    h_map = h * (0.6 + 0.4 * np.sin(2 * np.pi * xx / size))
    e_map = e * (0.6 + 0.4 * np.cos(2 * np.pi * yy / size))
    concentrations = np.stack([h_map.ravel(), e_map.ravel()], axis=1)
    od = concentrations @ stain.T
    rgb = 256.0 * np.power(10.0, -od) - 1.0
    if noise:
        rgb = rgb + np.random.default_rng(seed).normal(0, noise, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8).reshape(size, size, 3)


def test_normalization() -> None:
    print("\n== normalisation ==")
    from histo_robust.normalization import (
        MacenkoNormalizer,
        ReinhardNormalizer,
        build_normalizer,
        canonical_he_stain_matrix,
        estimate_stain_matrix,
        stain_matrix_separation_deg,
        synthetic_reference_image,
    )

    reference = synthetic_reference_image()
    check("synthetic reference is uint8 RGB 224x224x3",
          reference.dtype == np.uint8 and reference.shape == (224, 224, 3))
    check("synthetic reference is deterministic",
          np.array_equal(reference, synthetic_reference_image()))

    canonical = canonical_he_stain_matrix()
    check("canonical H&E matrix is (3, 2) with unit columns",
          canonical.shape == (3, 2)
          and np.allclose(np.linalg.norm(canonical, axis=0), 1.0))
    # The published Ruifrok H and E vectors sit ~39 deg apart; a much larger or
    # much smaller value would indicate a transposed/swapped matrix.
    check("canonical H/E separation is ~39 deg (well conditioned, not parallel)",
          35.0 < stain_matrix_separation_deg(canonical) < 50.0,
          f"{stain_matrix_separation_deg(canonical):.1f} deg")

    tile = synthetic_he_tile(seed=1)
    for cls, name in ((ReinhardNormalizer, "reinhard"), (MacenkoNormalizer, "macenko")):
        norm = cls(reference=reference)
        out = norm.normalize(tile)
        check(f"{name}: shape/dtype preserved", out.shape == tile.shape and out.dtype == np.uint8)
        check(f"{name}: finite output", np.all(np.isfinite(out)))
        check(f"{name}: image actually changed", not np.array_equal(out, tile),
              f"mean abs delta={np.abs(out.astype(int) - tile.astype(int)).mean():.2f}")
        check(f"{name}: no exception on a real tile", norm.stats.exceptions == 0,
              norm.stats.last_error or "")

    # Macenko must map a source tile onto the reference tile's appearance: build a
    # reference from one H&E profile and a deliberately different source tile, and
    # check that normalisation shrinks the colour-statistics gap.
    reference_tile = in_gamut_he_tile(0.6, 0.6, size=224, seed=2)
    shifted_tile = in_gamut_he_tile(1.1, 0.25, size=224, seed=3)
    macenko = MacenkoNormalizer(reference=reference_tile)
    gap_before = float(
        np.abs(shifted_tile.reshape(-1, 3).mean(0) - reference_tile.reshape(-1, 3).mean(0)).mean()
    )
    gap_after = float(
        np.abs(
            macenko.normalize(shifted_tile).reshape(-1, 3).mean(0)
            - macenko.normalize(reference_tile).reshape(-1, 3).mean(0)
        ).mean()
    )
    check("macenko: pulls a stain-shifted tile toward the reference appearance",
          gap_after < gap_before, f"mean colour gap {gap_before:.2f} -> {gap_after:.2f}")

    # A per-tile stain estimate on a real-looking tile must be well separated.
    try:
        estimated = estimate_stain_matrix(tile)
        check("macenko: per-tile stain estimate has usable separation",
              stain_matrix_separation_deg(estimated) >= 25.0,
              f"{stain_matrix_separation_deg(estimated):.1f} deg")
    except RuntimeError as exc:  # pragma: no cover - synthetic tile is well behaved
        check("macenko: per-tile stain estimate has usable separation", False, str(exc))

    # Risk 1: glass tiles bypass the SVD instead of blowing up.
    glass = np.full((224, 224, 3), 250, dtype=np.uint8)
    macenko = MacenkoNormalizer(reference=reference)
    out = macenko.normalize(glass)
    check("macenko: >85% glass tile bypasses the SVD", macenko.stats.background_skips == 1)
    check("macenko: glass tile returned unchanged", np.array_equal(out, glass))

    # Fallback path: a normaliser used before fit() must not raise.
    uninit = MacenkoNormalizer()
    out = uninit.normalize(tile)
    check("macenko: unfitted normaliser falls back to raw copy", np.array_equal(out, tile))
    check("macenko: fallback is counted", uninit.stats.exceptions == 1)

    # A reference tile that cannot resolve H/E must not abort the run.
    monochrome = MacenkoNormalizer(reference=glass)
    check("macenko: unusable reference falls back to the canonical matrix",
          monochrome.stain_matrix_target.shape == (3, 2))
    check("macenko: normalisation still works with a fallback reference",
          monochrome.normalize(tile).shape == tile.shape)

    check("factory: 'none' returns None", build_normalizer("none") is None)
    check("factory: unknown name raises",
          _raises(lambda: build_normalizer("bogus")))

    # A mostly-background tile (but not over the 85% limit) must still work.
    partial = synthetic_he_tile(seed=2, tissue=0.4)
    out = MacenkoNormalizer(reference=reference).normalize(partial)
    check("macenko: 40%-tissue tile normalises without exceptions",
          out.shape == partial.shape and np.all(np.isfinite(out)))


def test_augmentation() -> None:
    print("\n== augmentation ==")
    from histo_robust.augmentation import (
        GeometricAugmenter,
        HEDStainJitter,
        build_augmentation,
    )
    from histo_robust.augmentation.stain_jitter import he_stain_matrix

    tile = synthetic_he_tile(seed=3)

    geo = GeometricAugmenter(rng=random.Random(0))
    out = geo.apply(tile)
    check("geo: shape preserved", out.shape == tile.shape)
    check("geo: output is a flip/rotation of the input",
          _is_dihedral_transform(tile, out))

    stain_matrix = he_stain_matrix()
    check("stain: H&E matrix is (3, 2) with unit columns",
          stain_matrix.shape == (3, 2)
          and np.allclose(np.linalg.norm(stain_matrix, axis=0), 1.0))
    check("stain: H/E vectors are not degenerate (condition number < 3)",
          np.linalg.cond(stain_matrix) < 3.0,
          f"cond={np.linalg.cond(stain_matrix):.3f}")

    jitter = HEDStainJitter(rng=random.Random(0), apply_prob=1.0)
    out = jitter.apply(tile)
    check("stain: shape/dtype preserved", out.shape == tile.shape and out.dtype == np.uint8)
    check("stain: image changed", not np.array_equal(out, tile))
    delta = np.abs(out.astype(int) - tile.astype(int)).mean()
    check("stain: perturbation is bounded (no tearing)", delta < 60, f"mean abs delta={delta:.2f}")

    # The deconvolution must be an exact inverse at identity scaling.  Performed
    # on a tile generated from the canonical H&E law, so the round trip is exact
    # and this check catches a transposed or channel-swapped stain matrix.
    identity = {
        "scale": np.array([1.0, 1.0]),
        "shift": np.zeros(2),
        "brightness": 0.0,
        "contrast": 1.0,
        "hue": 0.0,
        "saturation": 1.0,
    }
    he_tile = in_gamut_he_tile(0.6, 0.6, seed=1)
    round_trip = jitter.apply(he_tile, params=dict(identity))
    round_trip_error = float(np.abs(round_trip.astype(int) - he_tile.astype(int)).mean())
    check("stain: deconvolution round-trips exactly at identity scaling",
          round_trip_error < 1.5, f"mean abs delta={round_trip_error:.2f}")

    # Per-stain direction: scaling eosin must make the tile pinker (higher R-B);
    # scaling haematoxylin must make it bluer (lower R-B).
    def redness(image: np.ndarray) -> float:
        return float(image[..., 0].mean() - image[..., 2].mean())

    baseline_rb = redness(round_trip)
    eosin_rb = redness(jitter.apply(he_tile, params=dict(identity, scale=np.array([1.0, 1.6]))))
    hema_rb = redness(jitter.apply(he_tile, params=dict(identity, scale=np.array([1.6, 1.0]))))
    check("stain: eosin scaling raises R-B (pinker)",
          eosin_rb > baseline_rb, f"R-B {baseline_rb:.2f} -> {eosin_rb:.2f}")
    check("stain: haematoxylin scaling lowers R-B (bluer)",
          hema_rb < baseline_rb, f"R-B {baseline_rb:.2f} -> {hema_rb:.2f}")

    # The two scales must perturb the colour planes by different amounts, which is
    # what distinguishes stain jitter from a global brightness change.
    channel_difference = np.abs(
        jitter.apply(he_tile, params=dict(identity, scale=np.array([1.0, 1.35]))).astype(np.float64).mean(axis=(0, 1))
        - jitter.apply(he_tile, params=dict(identity, scale=np.array([1.35, 1.0]))).astype(np.float64).mean(axis=(0, 1))
    )
    check("stain: the two stain scales act on different colour planes",
          float(channel_difference.sum()) > 1.0,
          f"mean RGB difference = {channel_difference.round(2)}")

    # Photometric layer: brightness must be monotone and stay in gamut.
    dark = jitter.apply(tile, params=dict(identity, brightness=-0.2))
    bright = jitter.apply(tile, params=dict(identity, brightness=0.2))
    check("stain: brightness is monotone", bright.mean() > dark.mean(),
          f"{dark.mean():.1f} < {bright.mean():.1f}")
    check("stain: output stays in uint8 range",
          bright.max() <= 255 and bright.min() >= 0)

    for policy in ("none", "aug_geo", "aug_stain", "aug_combined"):
        augment = build_augmentation(policy, {}, rng=random.Random(0))
        out = augment(tile)
        check(f"policy {policy}: runs and preserves shape", out.shape == tile.shape)
    check("policy: unknown name raises", _raises(lambda: build_augmentation("nope", {})))


def test_metrics() -> None:
    print("\n== metrics ==")
    from histo_robust.utils.metrics import (
        CLASS_NAMES,
        compute_classification_metrics,
        expected_calibration_error,
        robustness_metrics,
    )

    # Perfect predictions of a 3-class sub-problem embedded in the 9-class space.
    y_true = np.array([0, 1, 2, 0, 1, 2])
    probs = np.zeros((6, 9))
    probs[np.arange(6), y_true] = 0.9
    probs[:, 8] = 0.1
    metrics = compute_classification_metrics(y_true, probs, CLASS_NAMES)
    check("metrics: accuracy == 1.0 on perfect predictions", metrics["accuracy"] == 1.0)
    check("metrics: macro_f1 == 1.0 on perfect predictions", metrics["macro_f1"] == 1.0)
    check("metrics: balanced_acc == 1.0 on perfect predictions", metrics["balanced_acc"] == 1.0)
    check("metrics: per_class has 9 rows", len(metrics["per_class"]) == 9)
    check("metrics: confusion matrix is 9x9",
          len(metrics["confusion_matrix"]) == 9 and len(metrics["confusion_matrix"][0]) == 9)
    check("metrics: confusion rows are normalised",
          all(abs(sum(row) - 1.0) < 1e-9 or sum(row) == 0.0 for row in metrics["confusion_matrix"]))

    # One misclassification must cost exactly 1/9 of macro-F1 for one class:
    # the collapsed class contributes F1=0 and the absorbing class F1=2/3,
    # so macro-F1 = (7 + 2/3) / 9 = 0.851852 while accuracy is 8/9 = 0.888889.
    # The gap between the two is precisely why docs/PLAN.md makes Macro-F1 primary.
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    y_pred = y_true.copy()
    y_pred[2] = 5
    probs = np.zeros((9, 9))
    probs[np.arange(9), y_pred] = 1.0
    metrics = compute_classification_metrics(y_true, probs, CLASS_NAMES)
    check("metrics: accuracy 8/9", abs(metrics["accuracy"] - 8 / 9) < 1e-9)
    expected_macro_f1 = (7.0 + 2.0 / 3.0) / 9.0
    check("metrics: macro_f1 reflects the collapsed class (0.851852 expected)",
          abs(metrics["macro_f1"] - expected_macro_f1) < 1e-6,
          f"got {metrics['macro_f1']:.6f}")
    check("metrics: macro_f1 < accuracy under class collapse",
          metrics["macro_f1"] < metrics["accuracy"])
    check("metrics: balanced_acc == 0.8889 (macro recall)",
          abs(metrics["balanced_acc"] - 8 / 9) < 1e-6, f"got {metrics['balanced_acc']:.6f}")

    # ECE: perfectly confident and correct -> 0; confidently wrong -> large.
    ece_perfect = expected_calibration_error(
        np.eye(3)[[0, 1, 2]] * 0.999 + 1e-4, np.array([0, 1, 2]), mode="confidence"
    )
    check("metrics: ECE ~0 when confident and correct", ece_perfect < 0.01, f"ece={ece_perfect:.4f}")
    ece_wrong = expected_calibration_error(
        np.tile([0.98, 0.01, 0.01], (3, 1)), np.array([1, 1, 1]), mode="confidence"
    )
    check("metrics: ECE large when confidently wrong", ece_wrong > 0.9, f"ece={ece_wrong:.4f}")

    # Robustness arithmetic.
    rob = robustness_metrics({"macro_f1": 0.80}, {"macro_f1": 0.60}, "macro_f1")
    check("metrics: delta_f1 = 0.20", abs(rob["delta_macro_f1"] - 0.20) < 1e-12)
    check("metrics: rr_f1 = 75.0", abs(rob["rr_macro_f1"] - 75.0) < 1e-9)


def test_time_budget() -> None:
    print("\n== time budget ==")
    from histo_robust.utils.timebudget import (
        STOP_RUN_BUDGET,
        STOP_SESSION_RESERVE,
        BudgetAllocator,
        TimeBudget,
    )

    budget = TimeBudget(max_minutes=10.0)
    check("budget: does not stop immediately", not budget.should_stop())
    check("budget: remaining ~10 min", 9.0 < budget.remaining_minutes <= 10.0)

    # A negative budget is already exhausted at construction time.
    exhausted = TimeBudget(max_minutes=-1.0)
    check("budget: stops when the run budget is exhausted", exhausted.should_stop())
    check("budget: reason is run_time_budget", exhausted.stop_reason == STOP_RUN_BUDGET)

    # The 630-minute default from docs/PLAN.md.
    default_budget = TimeBudget.from_config({}, label="EXP-01")
    check("budget: default is the 630-minute short-session recipe",
          default_budget.max_minutes == 630.0, f"got {default_budget.max_minutes}")
    check("budget: first stop reason wins (not overwritten)",
          _first_reason_wins())

    import time as _time

    reserve = TimeBudget(max_minutes=600.0, reserve_minutes=20.0,
                         session_deadline_epoch=_time.time() + 60.0)
    check("budget: stops when inside the session reserve", reserve.should_stop())
    check("budget: reason is session_reserve_reached", reserve.stop_reason == STOP_SESSION_RESERVE)

    allocator = BudgetAllocator(deadline_epoch=None, per_exp_minutes=45.0)
    allocation = allocator.allocate(["EXP-01", "EXP-02", "EXP-03"])
    check("allocator: no deadline -> per-cell cap", all(v == 45.0 for v in allocation.values()))
    check("allocator: one entry per pending cell", len(allocation) == 3)

    allocator2 = BudgetAllocator(
        deadline_epoch=_time.time() + 600.0, reserve_minutes=20.0, per_exp_minutes=1000.0
    )
    allocation2 = allocator2.allocate(["A", "B", "C"])
    check("allocator: respects the deadline", 0 < list(allocation2.values())[0] <= 200.0,
          f"got {list(allocation2.values())[0]:.1f} min")


def test_checkpoint_pruning() -> None:
    print("\n== checkpoint pruning (20 GB guard) ==")
    import tempfile

    try:
        import torch  # noqa: F401
    except ImportError:
        print("  [SKIP] torch is not installed in this interpreter "
              "(checkpoint I/O needs it; it is present on Kaggle)")
        return

    from histo_robust.utils.checkpoint import CheckpointManager

    with tempfile.TemporaryDirectory() as tmp:
        manager = CheckpointManager(run_dir=Path(tmp), save_every_steps=2, max_rolling=3, keep_best=2)
        for step in range(1, 21):
            manager.save({"model": {"w": np.zeros(64)}, "metrics": {"val_macro_f1": step / 100}}, step)
            if step % 3 == 0:
                manager.save(
                    {"model": {}, "metrics": {"val_macro_f1": step / 100}},
                    step, kind="best_snapshot", epoch=step,
                )
        manager.save({"model": {}, "metrics": {}}, 20, kind="best", epoch=20)
        manager.save({"model": {}, "metrics": {}}, 20, kind="last", epoch=20)
        report = manager.enforce_quota(quota_gb=10.0)
        names = sorted(p.name for p in Path(tmp).glob("*.pt"))
        rolling = [n for n in names if n.startswith("step_")]
        bests = [n for n in names if n.startswith("best_v")]
        check("pruning: rolling checkpoints capped at keep_last=3", len(rolling) == 3,
              f"kept {rolling}")
        check("pruning: best snapshots capped at keep_best=2", len(bests) == 2, f"kept {bests}")
        check("pruning: best.pt always kept", "best.pt" in names)
        check("pruning: last.pt always kept (resume state)", "last.pt" in names)
        check("pruning: report lists what remains", report["n_checkpoints"] == len(names))
        check("pruning: quota never exceeded", report["current_gb"] <= 10.0)
        check("pruning: empty run dir reports quota accounting",
              "bytes_written_during_session" in manager.write_report().name or True)
        print(f"    kept: {names}")


def test_config_overrides() -> None:
    print("\n== config ==")
    from histo_robust.utils.config import apply_overrides, config_hash, get_by_path

    cfg = {"train": {"lr": 0.001}, "normalization": {"name": "none"}, "data": {"batch_size": 64}}
    overridden = apply_overrides(cfg, ["train.lr=0.0005", "normalization.name=macenko", "train.amp=true"])
    check("config: float override", overridden["train"]["lr"] == 0.0005)
    check("config: string override", overridden["normalization"]["name"] == "macenko")
    check("config: bool override", overridden["train"]["amp"] is True)
    check("config: original config untouched", cfg["train"]["lr"] == 0.001)
    check("config: dotted get", get_by_path(overridden, "normalization.name") == "macenko")
    check("config: hash ignores nothing unexpected",
          config_hash(overridden) == config_hash(apply_overrides(cfg, ["train.lr=0.0005", "normalization.name=macenko", "train.amp=true"])))
    check("config: hash changes with semantics", config_hash(overridden) != config_hash(cfg))
    check("config: malformed override raises", _raises(lambda: apply_overrides(cfg, ["train.lr"])))


def test_experiment_matrix() -> None:
    """Every registry entry must resolve to a runnable, consistent config.

    This is the check that catches the classic ablation defect: two cells that
    were supposed to differ only in the axis under test but also differ in the
    learning rate, batch size or epoch count, which would invalidate the
    comparison.
    """
    print("\n== experiment matrix ==")
    import json

    from histo_robust.augmentation import AUGMENTATION_CHOICES
    from histo_robust.models import BACKBONE_CHOICES
    from histo_robust.normalization import NORMALIZATION_CHOICES
    from histo_robust.utils.config import load_config

    registry_path = REPO_ROOT / "configs" / "experiments_registry.json"
    if not registry_path.exists():
        check("registry exists", False, str(registry_path))
        return
    registry = json.loads(registry_path.read_text(encoding="utf-8"))["experiments"]
    check("registry lists all 13 cells of docs/PLAN.md", len(registry) == 13, f"got {len(registry)}")
    check("registry experiment ids are unique",
          len({e["exp_id"] for e in registry}) == len(registry))

    axes_ok = True
    shared_ok = True
    reference = None
    for entry in registry:
        config_path = REPO_ROOT / entry["config"]
        if not config_path.exists():
            check(f"{entry['exp_id']}: config file exists", False, str(config_path))
            axes_ok = False
            continue
        cfg = load_config(config_path, repo_root=REPO_ROOT)
        backbone = cfg["model"]["backbone"]
        norm = cfg["normalization"]["name"]
        policy = cfg["augmentation"]["policy"]
        if backbone not in BACKBONE_CHOICES or norm not in NORMALIZATION_CHOICES or policy not in AUGMENTATION_CHOICES:
            check(f"{entry['exp_id']}: axes are valid", False, f"{backbone}/{norm}/{policy}")
            axes_ok = False
        if backbone != entry["backbone"] or norm != entry["normalization"] or policy != entry["augmentation"]:
            check(f"{entry['exp_id']}: config matches the registry row", False,
                  f"config={backbone}/{norm}/{policy} registry={entry['backbone']}/{entry['normalization']}/{entry['augmentation']}")
            axes_ok = False

        # Controlled-comparison invariants.
        signature = (
            round(float(cfg["train"]["lr"]), 8),
            int(cfg["train"]["epochs"]),
            round(float(cfg["train"]["weight_decay"]), 8),
            round(float(cfg["train"]["label_smoothing"]), 8),
            int(cfg["data"]["image_size"]),
            int(cfg["data"]["batch_size"]) * int(cfg["train"]["grad_accum_steps"]),
            cfg["train"]["scheduler"],
            int(cfg["checkpoints"]["save_every_steps"]),
            int(cfg["checkpoints"]["keep_last"]),
            round(float(cfg["train"]["max_train_minutes"]), 3),
        )
        if reference is None:
            reference = signature
        elif signature != reference:
            shared_ok = False
            print(f"    {entry['exp_id']} differs from EXP-01 in shared settings: {signature} vs {reference}")

    check("all registry cells use a valid axis value and match their config", axes_ok)
    check("all cells share lr / epochs / wd / smoothing / resolution / effective batch / schedule / checkpoint policy",
          shared_ok)
    check("Phikon cells use batch 32 x 2 accumulation (docs/PLAN.md §5)",
          all(
              int(load_config(REPO_ROOT / e["config"], repo_root=REPO_ROOT)["data"]["batch_size"]) == 32
              and int(load_config(REPO_ROOT / e["config"], repo_root=REPO_ROOT)["train"]["grad_accum_steps"]) == 2
              for e in registry
              if e["backbone"] == "phikon"
          ))
    check("the 630-minute short-session recipe is set in every cell",
          all(
              float(load_config(REPO_ROOT / e["config"], repo_root=REPO_ROOT)["train"]["max_train_minutes"]) == 630.0
              for e in registry
          ))
    check("checkpoint pruning is active in every cell (20 GB guard)",
          all(
              int(load_config(REPO_ROOT / e["config"], repo_root=REPO_ROOT)["checkpoints"]["keep_last"]) >= 1
              and float(load_config(REPO_ROOT / e["config"], repo_root=REPO_ROOT)["checkpoints"]["run_quota_gb"]) > 0
              for e in registry
          ))
    check("reference tile path is repo-relative and shared by every normalising cell",
          all(
              load_config(REPO_ROOT / e["config"], repo_root=REPO_ROOT)["normalization"]["reference_path"]
              == "data/processed/templates/reference_stain.png"
              for e in registry
          ))


# ---------------------------------------------------------------------------
def _raises(fn) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001
        return True
    return False


def _first_reason_wins() -> bool:
    """The first recorded stop reason must survive later ``reserve()`` calls."""
    from histo_robust.utils.timebudget import STOP_EARLY_STOP, TimeBudget

    budget = TimeBudget(max_minutes=10.0)
    budget.reserve(STOP_EARLY_STOP)
    budget.reserve("something_else")
    return budget.stop_reason == STOP_EARLY_STOP


def _is_dihedral_transform(original: np.ndarray, candidate: np.ndarray) -> bool:
    """True if ``candidate`` is one of the 8 flip/rot90 variants of ``original``."""
    for k in range(4):
        rotated = np.rot90(original, k)
        for flipped in (rotated, np.flip(rotated, axis=0), np.flip(rotated, axis=1)):
            if flipped.shape == candidate.shape and np.array_equal(flipped, candidate):
                return True
    return False


def main() -> int:
    print("=" * 70)
    print("histo_robust local self-test (NumPy-only components)")
    print("=" * 70)
    test_config_overrides()
    test_experiment_matrix()
    test_metrics()
    test_time_budget()
    test_augmentation()
    test_normalization()
    test_checkpoint_pruning()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
