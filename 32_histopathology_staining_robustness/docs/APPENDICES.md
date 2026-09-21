# Appendices — Reference

Reference material derived from the code in `../src/histo_robust/` and
`../scripts/`. Everything here is mechanically checkable against those files;
if this document and the code disagree, the code is right.

* [A. Output artifacts](#a-output-artifacts)
* [B. Configuration reference](#b-configuration-reference)
* [C. Command-line reference](#c-command-line-reference)
* [D. Kaggle constraints and guard rails](#d-kaggle-constraints-and-guard-rails)
* [E. Troubleshooting](#e-troubleshooting)
* [F. Glossary](#f-glossary)
* [G. Verification and self-tests](#g-verification-and-self-tests)

---

## A. Output artifacts

### A.1 Directory layout after a session

```text
/kaggle/working/
├── histo-robust/                     # extracted codebase (cell 1)
├── data/
│   └── processed/
│       ├── splits/                   # cell 3
│       │   ├── train.csv
│       │   ├── val_id.csv
│       │   ├── test_id.csv
│       │   ├── test_ood.csv
│       │   └── split_summary.json
│       ├── templates/reference_stain.png
│       └── cache/<method>/           # only if cell 4b was run
│           ├── train/ val_id/ test_id/ test_ood/
│           ├── *_cached.csv
│           └── cache_manifest.json
├── checkpoints/                      # cell 5
│   ├── time.json                     # per-session wall-clock accounting
│   └── <EXP-id>/
│       ├── best.pt                   # evergreen: best val_id Macro-F1
│       ├── last.pt                   # evergreen: resume state
│       ├── best_v####.pt             # <= keep_best snapshots
│       ├── step_########.pt          # <= keep_last rolling checkpoints
│       ├── train_report.json
│       └── checkpoint_report.json
├── results/
│   ├── logs/
│   │   ├── run_all_ablations.log
│   │   ├── train_<EXP-id>.log
│   │   ├── evaluate_<EXP-id>.log
│   │   └── preflight.log
│   ├── metrics/
│   │   ├── summary_results.csv        # the main results table
│   │   ├── ablation_manifest.json     # what is done / still pending
│   │   ├── session_summary.json       # one entry per session
│   │   └── preflight_benchmark.json
│   ├── per_experiment/<EXP-id>/
│   │   ├── evaluation_result.txt      # human-readable first stop
│   │   ├── evaluation_result.json
│   │   ├── metrics_test_id.json
│   │   ├── metrics_test_ood.json
│   │   ├── robustness.json
│   │   ├── predictions_test_id.npz
│   │   ├── predictions_test_ood.npz
│   │   ├── confusion_test_id.csv  /  .png
│   │   └── confusion_test_ood.csv /  .png
│   └── samples/                       # only with --save-samples
└── for_upload/
    └── histo-robust-checkpoints.zip   # re-upload as the checkpoint dataset
```

### A.2 `results/metrics/summary_results.csv`

The primary machine-readable table — one row per cell, written by
`scripts/train.py`, `scripts/evaluate.py` and `scripts/run_all_ablations.py`.
Column order is fixed by `histo_robust.engine.evaluator.SUMMARY_COLUMNS`.

| Column | Meaning |
| :--- | :--- |
| `exp_id` | `EXP-01` … `EXP-13` |
| `stage` | `Stage 0` … `Stage 4` (grouping from `docs/PLAN.md` §3) |
| `backbone` | `resnet50`, `convnext_tiny`, `phikon` |
| `weights_source` | `ImageNet-1k`, `Histopathology (TCGA iBOT)` |
| `normalization` | `none`, `reinhard`, `macenko` |
| `augmentation` | `none`, `aug_geo`, `aug_stain`, `aug_combined` |
| `status` | `completed`, `interrupted` (time-boxed), `failed` |
| `stop_reason` | `max_epochs`, `early_stopping`, `run_time_budget`, `session_reserve_reached`, `cuda_oom` |
| `epochs_completed` | epochs actually finished |
| `best_val_macro_f1` | peak `val_id` Macro-F1 used for checkpoint selection |
| `best_epoch` | epoch at which that peak occurred |
| `trained_minutes` | wall-clock minutes spent training |
| `test_id_accuracy`, `test_id_balanced_acc`, `test_id_macro_f1`, `test_id_macro_auroc`, `test_id_ece` | in-domain held-out metrics |
| `test_ood_*` | the same five metrics on `CRC-VAL-HE-7K` |
| `delta_f1`, `rr_f1` | `F1_ID − F1_OOD`; `F1_OOD / F1_ID × 100` |
| `delta_balanced_acc`, `rr_balanced_acc` | same pair for balanced accuracy |
| `delta_accuracy`, `rr_accuracy` | same pair for top-1 accuracy |
| `delta_auroc`, `rr_auroc` | same pair for macro AUROC |
| `config_hash` | 12-hex-char hash of the semantic config (provenance) |
| `checkpoint` | path of the checkpoint that produced these numbers |
| `notes` | free text; records the stop reason and stain-reference provenance |

A cell with `status = interrupted` has **interim** numbers: training was cut by
the time box, the next session resumes it, and the row is overwritten. Do not
treat an interrupted row as a final result.

### A.3 `results/metrics/ablation_manifest.json`

Durable record of progress; this is what lets a new session skip finished cells.

```jsonc
{
  "created_utc": "…",
  "sessions": [{ "started_utc": "…", "pid": 123, "host": "…", "…": "summary fields" }],
  "experiments": {
    "EXP-01": {
      "status": "completed",              // completed | interrupted | failed
      "stop_reason": "max_epochs",
      "epochs_completed": 12,
      "planned_epochs": 12,
      "best_val_macro_f1": 0.0,           // pending
      "best_epoch": -1,
      "trained_seconds": 0.0,
      "best_path": "/kaggle/working/checkpoints/EXP-01/best.pt",
      "last_path": "/kaggle/working/checkpoints/EXP-01/last.pt",
      "stage": "Stage 0",
      "config": "configs/experiments/exp01_baseline_resnet50.yaml",
      "config_hash": "…",
      "batch_size": 64,
      "test_id_macro_f1": null,
      "test_ood_macro_f1": null,
      "delta_f1": null,
      "rr_f1": null,
      "evaluation": { "delta_macro_f1": null, "rr_macro_f1": null },
      "attempt": 1,
      "updated_utc": "…"
    }
  },
  "session_summary": { "session_minutes": 720, "total_trained_minutes": 0.0 }
}
```

The literal `null`s above are placeholders for the shape — a real run fills them
with numbers. A cell counts as **done** only when `status == "completed"` **and**
`epochs_completed >= planned_epochs` (or it genuinely early-stopped).

### A.4 `metrics_test_id.json` / `metrics_test_ood.json`

Full metric bundle for one split, produced by
`histo_robust.utils.metrics.compute_classification_metrics`.

| Field | Definition |
| :--- | :--- |
| `n_samples` | rows evaluated |
| `accuracy` | overall top-1 accuracy |
| `balanced_acc` | macro-averaged recall |
| `macro_f1` | harmonic mean of precision/recall per class, averaged over the classes present |
| `weighted_f1` | support-weighted F1 (context only) |
| `macro_auroc` | macro one-vs-rest AUROC on softmax probabilities; `NaN` if a class is absent |
| `ece` | expected calibration error, 15 bins, confidence-based |
| `ece_label` | ECE binned by the probability of the true class (less saturated) |
| `mean_confidence` | mean max softmax probability |
| `per_class[]` | 9 entries: `class_index`, `class_name`, `precision`, `recall`, `f1`, `auroc`, `support` |
| `confusion_matrix` | 9×9 row-normalised (rows = true class) |

### A.5 `robustness.json`

```jsonc
{
  "exp_id": "EXP-03",
  "config_hash": "…",
  "macro_f1_id": 0.0, "macro_f1_ood": 0.0,
  "delta_macro_f1": 0.0, "rr_macro_f1": 0.0,
  "balanced_acc_id": 0.0, "balanced_acc_ood": 0.0,
  "delta_balanced_acc": 0.0, "rr_balanced_acc": 0.0,
  "accuracy_id": 0.0, "accuracy_ood": 0.0,
  "delta_accuracy": 0.0, "rr_accuracy": 0.0,
  "macro_auroc_id": 0.0, "macro_auroc_ood": 0.0,
  "delta_macro_auroc": 0.0, "rr_macro_auroc": 0.0,
  "interpretation": { "delta": "…", "rr": "…" },
  "stage": "Stage 1", "backbone": "resnet50",
  "normalization": "macenko", "augmentation": "none",
  "attempt": 1, "run_status": "completed", "epochs_completed": 12, "planned_epochs": 12
}
```

### A.6 `predictions_test_{id,ood}.npz`

Compressed arrays for independent re-derivation of every reported number:

| Array | Shape | Meaning |
| :--- | :--- | :--- |
| `y_true` | `(N,)` int16 | ground-truth class index |
| `probs` | `(N, 9)` float32 | softmax probabilities |
| `indices` | `(N,)` int64 | row index into the split CSV |
| `paths` | `(N,)` object | resolved image path |

Recomputing a metric from these is the audit path used by the verification
stage; see [§G](#g-verification-and-self-tests).

### A.7 `checkpoints/<EXP>/train_report.json`

Provenance for one training run: `config_path`, `config_hash`, `seed`,
`hostname`, `platform`, `train_result` (status, stop reason, epochs, best
metric, checkpoint paths, `checkpoint_report`), `train_history`, `val_history`,
`time_budget`, `session_clock`, `environment` (GPU names and memory), effective
config, `reference_provenance`, `cached_splits_dir`, `checkpoints_size_gb`,
`disk_free_gb`, `wall_minutes`, and — when the post-training audit ran —
`evaluation`.

### A.8 `checkpoint_report.json`

Proof that the 20 GB guard did its job: `quota_gb`, `current_gb`,
`n_checkpoints`, `files[]` (name, kind, size, step, epoch), `removed_now[]`,
`save_every_steps`, `max_rolling_keep_last`, `keep_best`,
`bytes_written_during_session`.

### A.9 `split_summary.json`

Split provenance and the firewall evidence: `source_root`, `target_root`,
`subset` (`subset25000` or `full`), `seed`, `split_fractions`,
`n_source_available`, `n_source_used`, per-split `n_samples` / `domains` /
`class_counts`, `csv_sha256` for each split CSV, and
`reference_source_image` — the training tile chosen as the canonical stain
reference.

### A.10 `checkpoints/time.json`

Session accounting across resumptions: `session_minutes`,
`total_trained_seconds`, and `experiments.<EXP>.trained_seconds`, so a resumed
session knows how long a cell has already consumed and the budget allocator does
not restart its accounting from zero.

### A.11 Checkpoint payload (`*.pt`)

```jsonc
{
  "format": "histo_robust/v1",
  "exp_id": "EXP-03",
  "epoch": 7,
  "global_step": 2184,
  "model": { "…": "state_dict" },
  "optimizer": { "…": "state_dict" },
  "scheduler": { "…": "state_dict" },
  "scaler": { "…": "AMP GradScaler state" },
  "best_val_macro_f1": 0.0,
  "best_epoch": -1,
  "rng": { "python": "…", "numpy": "…", "torch": "…", "cuda": [ "…" ] },
  "config_hash": "…",
  "config": { "…": "effective config, underscore-keys stripped" },
  "metrics": { "…": "validation metrics for this epoch" },
  "trained_seconds": 0.0,
  "timestamp": "…"
}
```

`epoch` is the epoch that **completed**, so resume starts at `epoch + 1`. The
`rng` block is what makes a resumed run continue the same random stream. Writes
are atomic (`*.tmp` then `os.replace`), which is why a session killed mid-write
cannot leave a corrupt checkpoint.

---

## B. Configuration reference

Configs live in `../configs/`: `base_config.yaml` holds the defaults,
`experiments/exp*.yaml` are the 13 fully-expanded per-cell files (generated by
`configs/_generate_experiment_configs.py`), and `experiments_registry.json` is
the single source of truth for matrix ordering.

Any key can be overridden from the command line with a dotted path:

```bash
python scripts/train.py --config configs/experiments/exp01_baseline_resnet50.yaml \
    --set data.batch_size=32 --set model.freeze_backbone=true
```

### B.1 `runtime` and `paths`

| Key | Default | Notes |
| :--- | :--- | :--- |
| `runtime.seed` | `42` | seeds Python/NumPy/torch and cuDNN determinism |
| `runtime.deterministic` | `true` | cuDNN deterministic mode |
| `paths.data_root` | `data/raw` | raw dataset location |
| `paths.splits_dir` | `data/processed/splits` | split CSVs; overridden when a cache is adopted |
| `paths.normalization_cache` | `null` | set to `data/processed/cache/<method>` to read pre-normalised tiles |
| `paths.cache_dir` | `data/processed/cache` | cache output root |
| `paths.results_dir` / `paths.checkpoints_dir` | `results` / `checkpoints` | output roots |
| `paths.weights_dir` | `null` | offline pretrained-weights directory (Internet OFF) |
| `paths.search_roots` | `[]` | extra roots for resolving split CSV `image_path` values |

### B.2 `data`

| Key | Default | Notes |
| :--- | :--- | :--- |
| `class_names` | `[ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM]` | **positional**; never reorder |
| `num_classes` | `9` | |
| `image_size` | `224` | native patch resolution; no upsampling |
| `batch_size` | `64` | 32 for the Phikon cells |
| `num_workers` | `4` | matches Kaggle's vCPU count |
| `pin_memory` | `true` | |
| `persistent_workers` | `true` | |
| `drop_last` | `false` | keeps every training patch |

> `data.batch_size` and `train.batch_size` are aliases; `--batch-size` sets both.

### B.3 `normalization` (Axis B)

| Key | Default | Allowed / notes |
| :--- | :--- | :--- |
| `name` | `none` | `none` \| `reinhard` \| `macenko` |
| `reference_path` | `data/processed/templates/reference_stain.png` | repo-relative; resolved against the repo root |
| `params.use_tissue_mask` | `true` | Reinhard: fit statistics on tissue only |
| `params.luminance_threshold` | `220` | Reinhard tissue/glass cut |
| `params.od_threshold` | `0.15` | Macenko: minimum optical-density norm |
| `params.angular_percentile` | `99.0` | Macenko: extreme-azimuth percentile |
| `params.concentration_percentile` | `99.0` | Macenko: dynamic-range percentile |
| `params.alpha` | `1.0` | Macenko: concentration scale factor |
| `params.background_fraction_limit` | `0.85` | Macenko: above this glass fraction, bypass the SVD |

Both methods operate on uint8 RGB tiles and return uint8 RGB of the same shape,
are stateless after fit (safe inside DataLoader workers), and never raise on a
bad tile — failures are counted and demoted to a raw copy.

### B.4 `augmentation` (Axis C)

| Key | Default | Allowed / notes |
| :--- | :--- | :--- |
| `policy` | `none` | `none` \| `aug_geo` \| `aug_stain` \| `aug_combined` |
| `resize` | `null` | optional pre-crop resize; `null` keeps native 224 |
| `geometric.hflip_prob` | `0.5` | |
| `geometric.vflip_prob` | `0.5` | |
| `geometric.rot90_prob` | `0.5` | random 90° rotation, k ∈ {1,2,3} |
| `stain.he_scale` | `[0.85, 1.15]` | per-stain concentration scaling |
| `stain.he_shift` | `[-0.08, 0.08]` | per-stain offset, as a fraction of the canonical concentration |
| `stain.brightness` | `0.10` | global gain |
| `stain.contrast` | `0.10` | global contrast |
| `stain.hue` | `0.03` | HSV hue shift |
| `stain.saturation` | `0.12` | HSV saturation scale |
| `stain.apply_prob` | `0.8` | probability the stain jitter fires |
| `stain.use_dab_channel` | `false` | DAB is an IHC counterstain; off for H&E |

Order is fixed and shared by training, evaluation and the cache: **normalise →
center-crop → augment → tensor + ImageNet normalisation**.

### B.5 `model` (Axis E)

| Key | Default | Allowed / notes |
| :--- | :--- | :--- |
| `backbone` | `resnet50` | `resnet50` (`resnet50.a1_in1k`) \| `convnext_tiny` (`convnext_tiny.fb_in1k`) \| `phikon` (`owkin/phikon`) |
| `pretrained` | `true` | |
| `weights_source` | `ImageNet-1k` | provenance label only |
| `dropout` | `0.1` | `0.0` for the linear-probe Phikon cells |
| `freeze_backbone` | `false` | `true` → cached-feature linear probe |

### B.6 `train`

| Key | Default | Notes |
| :--- | :--- | :--- |
| `epochs` | `12` | |
| `lr` | `0.001` | AdamW |
| `head_lr_mult` | `1.0` | separate LR multiplier for the head |
| `weight_decay` | `0.05` | |
| `betas` | `[0.9, 0.999]` | |
| `label_smoothing` | `0.1` | |
| `grad_accum_steps` | `1` | `2` for Phikon |
| `grad_clip_norm` | `1.0` | |
| `amp` | `true` | fp16 autocast + GradScaler |
| `scheduler` | `cosine` | `cosine` \| `step` \| `none` |
| `warmup_epochs` | `1.0` | |
| `min_lr_ratio` | `0.01` | cosine floor |
| `early_stopping_patience` | `0` | `0` disables (cells are time-boxed, not plateau-boxed) |
| `max_train_minutes` | `630` | **the 10.5 h short-session recipe** |
| `oom_retry` / `oom_batch_floor` | `true` / `8` | OOM retry ladder floor |

### B.7 `session` and `checkpoints`

| Key | Default | Notes |
| :--- | :--- | :--- |
| `session.session_minutes` | `720` | Kaggle's hard kill |
| `session.reserve_minutes` | `20` | held back for flush + zipping |
| `session.session_start` | `null` | `"YYYY-MM-DD HH:MM:SS"` UTC; also read from `KAGGLE_SESSION_START` |
| `checkpoints.save_every_steps` | `2000` | `--save-every` |
| `checkpoints.keep_last` | `3` | `--keep-last` |
| `checkpoints.keep_best` | `3` | `--keep-best` |
| `checkpoints.run_quota_gb` | `3.0` | per-cell ceiling enforced by pruning |

### B.8 `evaluation`

| Key | Default | Notes |
| :--- | :--- | :--- |
| `primary_metric` | `macro_f1` | |
| `metrics` | `[accuracy, balanced_acc, macro_f1, macro_auroc, ece]` | |
| `ece_bins` | `15` | |
| `robustness.absolute_drop` | formula string | `delta = macro_f1_test_id − macro_f1_test_ood` |
| `robustness.retention_rate` | formula string | `rr = macro_f1_test_ood / macro_f1_test_id × 100` |

### B.9 Configs are self-contained

Each `configs/experiments/*.yaml` writes out the **full merged config** rather
than a diff, so a reviewer can compare two cells directly and what you read is
what runs. `_base_source_` and `_overrides_` record provenance only and are not
consumed by the loader.

---

## C. Command-line reference

All scripts set up `sys.path` themselves, so they run from the repository root
without installation. `pip install -e .` additionally exposes
`histo-robust-train`, `histo-robust-evaluate`, `histo-robust-splits` and
`histo-robust-smoke`.

### C.1 `scripts/prepare_splits.py`

Builds `train.csv`, `val_id.csv`, `test_id.csv`, `test_ood.csv` plus
`split_summary.json`, and asserts the leakage firewall.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--data-root` | *required* | folder containing the nine source class folders |
| `--target-root` | `<data-root>/../CRC-VAL-HE-7K` | folder containing the nine target class folders |
| `--out-dir` | `data/processed/splits` | where the CSVs go |
| `--subset` | `25000` | stratified source subset size |
| `--full` | off | use all 100,000 source patches (ignores `--subset`) |
| `--seed` | `42` | split seed |
| `--reference-out` | none | copy the chosen canonical reference tile here |
| `--limit-target` | `0` | cap `test_ood` rows (0 = all) |

### C.2 `scripts/preprocess_normalize.py`

Optional offline normalisation cache. Writes tiles plus `<split>_cached.csv`
files and `cache_manifest.json`.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--config` | none | experiment YAML supplying `normalization` params |
| `--normalization` | from config | `reinhard` \| `macenko` |
| `--splits-dir` / `--out-dir` | `data/processed/splits` / `data/processed/cache` | input / output |
| `--reference` | resolved | canonical reference tile |
| `--image-size` | from config (`224`) | crop size |
| `--workers` | `4` | parallel processes |
| `--splits` | all four | subset of splits to cache |
| `--limit` | `0` | debug: first N rows only |
| `--search-root` | none (repeatable) | extra roots for resolving `image_path` |
| `--no-cached-splits` | off | do not emit `<split>_cached.csv` |
| `--set` | none (repeatable) | dotted config overrides |

### C.3 `scripts/smoke_test.py`

Pre-flight benchmark: structural split/firewall checks, DataLoader throughput, a
real forward/backward timing loop, extrapolated minutes-per-epoch, and a
`OK` / `CPU-BOUND` / `IDLE-GPU` verdict.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--config` | *required* | config to benchmark |
| `--steps` / `--warmup-steps` | `10` / `2` | timed steps, excluded warm-up |
| `--batch-size` | from config | override |
| `--skip-gpu-assert` | off | allow a CPU run (timings will not extrapolate) |
| `--output` | `results/metrics/preflight_benchmark.json` | report path |
| `--save-samples` / `--samples-dir` | `0` / `results/samples` | dump raw/normalised/augmented tiles |
| `--weights-dir`, `--reference`, `--search-root`, `--set` | — | as elsewhere |

### C.4 `scripts/train.py`

Train one cell; auto-resume; time-boxed; optional post-training audit.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--config` | *required* | experiment YAML |
| `--exp-id` | config `exp_id` / stem | checkpoint and report folder name |
| `--output-dir` / `--results-dir` | `checkpoints` / `results` | output roots |
| `--log-dir` | `<results>/logs` | log location |
| `--max-minutes` | config `630` | training budget for this run |
| `--session-start` / `--reserve-minutes` | env / `20` | absolute deadline and reserve |
| `--save-every` / `--keep-last` / `--keep-best` / `--run-quota-gb` | config `2000` / `3` / `3` / `3.0` | checkpoint policy |
| `--epochs` / `--batch-size` / `--lr` | config | overrides |
| `--limit-train` / `--limit-val` / `--limit-test-id` | `0` | debug row caps |
| `--smoke` | off | tiny run (1 epoch, capped rows, small budget) |
| `--no-resume` | off | ignore existing checkpoints |
| `--no-amp` | off | disable AMP (debugging only) |
| `--eval-test-id` / `--eval-test-ood` | off | post-training audits |
| `--save-samples` | `0` | sample tiles per split |
| `--weights-dir` / `--reference` / `--normalization-cache` / `--search-root` / `--set` | — | as elsewhere |

### C.5 `scripts/evaluate.py`

Post-training audit only. Loads the frozen `best.pt` (falling back to `last.pt`)
and writes the full artifact set.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--config` | *required* | must match the training config |
| `--exp-id` | config `exp_id` | which cell |
| `--checkpoint-dir` | `checkpoints` | root containing `<exp-id>/best.pt` |
| `--results-dir` | `results` | output root |
| `--limit-test-id` / `--limit-test-ood` | `0` | debug row caps |
| `--n-bootstrap`, `--save-samples` | `0` | reserved; currently no-ops |
| `--no-amp`, `--reference`, `--weights-dir`, `--normalization-cache`, `--search-root`, `--set` | — | as elsewhere |

### C.6 `scripts/run_all_ablations.py`

Session orchestrator.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--registry` | `configs/experiments_registry.json` | matrix source of truth |
| `--checkpoint-dir` / `--results-dir` / `--log-dir` | `checkpoints` / `results` / `<results>/logs` | output roots |
| `--experiments` | all | subset of ids, e.g. `--experiments EXP-01 EXP-03` |
| `--per-exp-minutes` | `75` | per-cell cap |
| `--min-minutes` | `6` | do not start a cell with less left |
| `--session-minutes` / `--reserve-minutes` / `--session-start` | `700` / `20` / env | window |
| `--max-total-minutes` | none | hard cap on this orchestrator's wall clock |
| `--eval-test-ood` | off | run the OOD audit after each cell |
| `--force` / `--no-resume` / `--dry-run` | off | re-run completed cells / never resume / print the plan only |
| `--oom-retries` | `2` | retries per cell with halved batch |
| `--batch-size`, `--epochs`, `--save-every`, `--keep-last`, `--weights-dir`, `--reference`, `--normalization-cache`, `--search-root`, `--set` | — | passed through to each cell |

### C.7 `scripts/make_zips.py`

Builds the upload archives with guaranteed forward-slash entry names and verifies
each archive afterwards.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--out-dir` | `dist` | where the zips are written |
| `--data-dir` | `zipped_dataset` | existing Zenodo archives to copy |
| `--checkpoints` | none | directory to bundle for re-upload |
| `--repack-data` / `--raw-root` | off / `data/raw` | re-create the dataset zips from raw folders |
| `--zip-name-version` | none | suffix, e.g. `v2` |
| `--repo-root` / `--include-code` | repo root / on | codebase archive control |

---

## D. Kaggle constraints and guard rails

| Constraint | Limit | Guard in code | Where |
| :--- | :--- | :--- | :--- |
| Session length | 12 h hard kill | stop at `max_train_minutes = 630`; also stop `reserve_minutes = 20` before the session deadline | `utils/timebudget.py`, `engine/trainer.py` |
| Output size | 20 GB on `/kaggle/working` | `save_every_steps` + `keep_last` + `keep_best` + per-cell `run_quota_gb` pruning; raw images read from read-only `/kaggle/input` | `utils/checkpoint.py` |
| Upload format | folders break the web uploader | three forward-slash-verified zips | `scripts/make_zips.py` |
| Statelessness | no disk between sessions | resume from the newest `last.pt` under `/kaggle/input`; manifest skips finished cells | `scripts/run_all_ablations.py`, cell 2 |
| Accelerator | GPU required | hard assertion in cell 1 | notebook cell 1 |
| VRAM | 16 GB P100 / T4 | AMP, `grad_accum_steps` for Phikon, `freeze_backbone` linear probe, automatic OOM retry with halved batch | `engine/trainer.py`, `scripts/run_all_ablations.py` |

Budget arithmetic with the shipped defaults: 630 min ÷ 13 cells ≈ 48 min per
cell in a single uninterrupted session; the orchestrator's `--per-exp-minutes 75`
cap and 700-minute window mean **two or three sessions for the full matrix**,
with every cell's progress carried forward.

---

## E. Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `FATAL: could not find the codebase zip` | codebase dataset not attached, or it holds a folder rather than a zip | attach `histo-robust-code`; recreate it from `histo-robust-code.zip` |
| `FATAL: could not locate the datasets` / `missing class folders` | dataset uploaded unzipped-and-flattened, or zipped with backslashes | re-upload the original archive (or one built by `make_zips.py`) |
| `No CUDA device is available` | accelerator not set before the kernel started | Settings → Accelerator → `GPU P100`/`T4 x2`, then re-run cell 1 |
| `CUDA out of memory` | batch too large for the card | retried automatically with halved batch; force with `--batch-size 32` |
| `Couldn't reach huggingface.co` | Internet OFF, or a throttled download | enable Internet, or mount weights and set `--weights-dir` (guide §7) |
| Run ended after ~10.5 h, status `interrupted` | by design — `run_time_budget` fired | download outputs, update the checkpoint dataset, re-run |
| `summary_results.csv` missing cells | those cells had not finished | expected in multi-session runs; `ablation_manifest.json` lists what remains |
| Val Macro-F1 well below expectation | few epochs ran (time-boxed), or LR/batch overridden | check minutes-per-epoch from cell 4 and the per-cell budget in the log |
| `macenko` fallbacks non-zero | a tile could not resolve H/E | expected for `BACK`; investigate if large for tissue-dense classes (see `RESULTS.md` Table 6) |
| Results folder missing after a session | outputs not downloaded before recycling | download immediately; the checkpoint zip is the resumable state |

---

## F. Glossary

| Term | Meaning |
| :--- | :--- |
| **ID** | in-domain — `test_id`, the held-out 15 % of the source cohort |
| **OOD** | out-of-domain — `test_ood`, the full `CRC-VAL-HE-7K` cohort |
| **Δ (delta)** | `ID − OOD` for a metric; **lower is better** |
| **RR** | retention rate `OOD / ID × 100`; **higher is better** |
| **Macro-F1** | per-class F1 averaged over classes; robust to class imbalance |
| **Bal-Acc** | balanced accuracy, the macro-average of recall |
| **AUROC** | macro one-vs-rest area under the ROC curve over softmax probabilities |
| **ECE** | expected calibration error over 15 confidence bins |
| **H&E** | haematoxylin and eosin, the standard histological stain pair |
| **OD** | optical density, `−log10((I+1)/256)` — the Beer-Lambert domain where stains add linearly |
| **Reinhard** | statistical colour transfer in CIELAB (mean/std matching) |
| **Macenko** | optical-density H&E deconvolution to stain vectors, then concentration normalisation |
| **HED jitter** | augmentation that rescales/shifts H and E concentrations independently and rebuilds the tile |
| **Aug-Geo** | flips + random 90° rotations only |
| **Aug-Stain** | H&E stain jitter + brightness/contrast + HSV perturbation |
| **Aug-Combined** | Aug-Geo ∘ Aug-Stain |
| **linear probe** | frozen backbone with only the classification head trained (cached features) |
| **cell** | one row of the ablation matrix, i.e. one `EXP-xx` configuration |
| **session** | one Kaggle background run, terminated at 12 h |
| **firewall** | the rule that `CRC-VAL-HE-7K` never influences training or model selection |
| **manifest** | `results/metrics/ablation_manifest.json`, the durable progress record |

---

## G. Verification and self-tests

### G.1 Dependency-light self-test

```bash
python tests/local_selftest.py
```

Runs without torch/pandas and covers: config loading and dotted overrides,
metric correctness on hand-computed cases, `delta`/`rr` arithmetic, the time
budget's stop conditions and reason precedence, geometric and stain
augmentation invariants, Reinhard/Macenko stability plus the glass-tile and
unfitted-normaliser fallbacks, checkpoint pruning (`keep_last`, `keep_best`,
`best.pt`/`last.pt` retention, quota enforcement), and an **experiment-matrix
audit** asserting that all 13 cells use valid axis values, match
`experiments_registry.json`, and share lr / epochs / weight decay / smoothing /
resolution / effective batch / schedule / checkpoint policy.

### G.2 Re-deriving a reported number

The audit path for any result in `docs/RESULTS.md`:

```python
import numpy as np
from histo_robust.utils.metrics import CLASS_NAMES, compute_classification_metrics

data = np.load("results/per_experiment/EXP-03/predictions_test_ood.npz", allow_pickle=True)
metrics = compute_classification_metrics(data["y_true"], data["probs"], CLASS_NAMES)
print(metrics["macro_f1"], metrics["balanced_acc"], metrics["accuracy"], metrics["macro_auroc"])
```

This must reproduce `metrics_test_ood.json` exactly. `confusion_test_ood.csv` can
be cross-checked the same way (row-normalised, rows = true class).

### G.3 Firewall assertions that already run for you

* `prepare_splits.py` fails if any image path appears in two splits, if
  `train`/`val_id` contain anything but `source`, or if `test_ood` contains
  anything but `target`.
* `histo_robust.data.datamodule` refuses to build `test_ood` unless
  `post_training=True`, and exposes no way for a training loop to obtain it.
* `smoke_test.py` re-checks both conditions before a 10-hour run starts.
* Checkpoints are selected on `val_id` Macro-F1 only — never on an OOD metric.
