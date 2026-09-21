# Robust Histopathology Image Classification under Staining Variations

Binary-free, single-GPU study of **what actually makes a histopathology classifier
robust to staining variation**: colour normalisation, augmentation policy, and
pretrained backbone choice, evaluated as an in-domain → out-of-domain transfer
problem.

* **Source (in-domain)**: `NCT-CRC-HE-100K-NONORM` — 100,000 raw H&E patches,
  Heidelberg/Mannheim, 9 colorectal tissue classes, 224×224.
* **Target (out-of-domain)**: `CRC-VAL-HE-7K` — 7,180 patches from 50 independent
  RWTH Aachen patients. Never used for training, scheduling, or checkpoint
  selection.

Planning documents: [`PLAN.md`](PLAN.md) (experiment design, evaluation protocol,
tech stack) and [`dataset_card.md`](dataset_card.md) (dataset, splits, folder
layout).

## Quick start on Kaggle

Full procedure: [`kaggle_guide.md`](kaggle_guide.md).

```powershell
# 1. Build the three upload artifacts (forward-slash-safe zips; never use the
#    Windows "Send to > Compressed folder" tool)
python scripts/make_zips.py --out-dir dist

# 2. Upload as three private Kaggle datasets:
#    histo-robust-code, nct-crc-he-100k-nonorm, crc-val-he-7k

# 3. Import kaggle/notebook_01_run_all_ablations.ipynb, attach the datasets,
#    set Accelerator = GPU P100 (or T4 x2) and Internet = On, run cells 1-4,
#    then Save Version -> Save & Run All (Commit).
```

## Local (CPU or single GPU)

```bash
pip install -e .

python scripts/prepare_splits.py \
    --data-root data/raw/NCT-CRC-HE-100K-NONORM \
    --target-root data/raw/CRC-VAL-HE-7K \
    --out-dir data/processed/splits --subset 25000 --seed 42 \
    --reference-out data/processed/templates/reference_stain.png

python scripts/smoke_test.py --config configs/experiments/exp01_baseline_resnet50.yaml

python scripts/train.py --config configs/experiments/exp01_baseline_resnet50.yaml \
    --exp-id EXP-01 --output-dir checkpoints --results-dir results \
    --eval-test-id --eval-test-ood

python scripts/run_all_ablations.py --checkpoint-dir checkpoints --results-dir results \
    --per-exp-minutes 75 --eval-test-ood

python tests/local_selftest.py     # ~30 s, dependency-light sanity checks
```

## The 13-cell ablation matrix

| Axis | Cells | Question |
| :--- | :--- | :--- |
| A — anchor | EXP-01 | Raw RGB, ResNet-50, no augmentation |
| B — normalisation | EXP-02, EXP-03 | Reinhard vs Macenko vs none |
| C — augmentation | EXP-04, EXP-05, EXP-06 | geometric vs stain vs combined |
| D — interaction | EXP-07, EXP-08, EXP-09 | Macenko × each augmentation policy |
| E — backbone | EXP-10, EXP-11, EXP-12, EXP-13 | ConvNeXt-Tiny and Phikon (histology SSL) under raw and defended settings |

`configs/experiments_registry.json` is the single source of truth for the matrix;
`scripts/run_all_ablations.py`, `RUN_LOG.md` and `RESULTS.md` all follow its order.

## Evaluation

Primary metric **Macro-F1** (the OOD cohort is imbalanced: `ADI` 1,338 vs `DEB`
339). Also reported: balanced accuracy, top-1 accuracy, macro one-vs-rest AUROC,
expected calibration error, and full 9×9 row-normalised confusion matrices.

Robustness is quantified as `ΔF1 = F1_ID − F1_OOD` (lower is better) and
`RR-F1 = F1_OOD / F1_ID × 100` (higher is better).

## Repository layout

```text
├── PLAN.md / dataset_card.md      # planning documents (read first)
├── kaggle_guide.md                # click-by-click Kaggle procedure
├── RUN_LOG.md / RESULTS.md        # run register and results tables (to be filled)
├── configs/                       # base_config.yaml + experiments_registry.json
│   └── experiments/               # 13 generated per-cell YAMLs
├── src/histo_robust/              # library: normalization, augmentation, data,
│                                  # models, engine, utils
├── scripts/                       # prepare_splits, preprocess_normalize, smoke_test,
│                                  # train, evaluate, run_all_ablations, make_zips
├── kaggle/                        # notebooks + notebook_helpers
└── tests/local_selftest.py        # dependency-light self-test suite
```

## Kaggle survival features

* **12 h hard kill** — training stops itself at 630 min (10.5 h) and also reserves
  20 min before the session deadline, writing `last.pt` (model + optimizer +
  scheduler + AMP scaler + RNG) and `best.pt` before returning a clean exit code.
* **20 GB output cap** — `--save-every` + `--keep-last 3` rolling retention,
  `--keep-best` snapshots, and a per-cell GB quota enforced by auto-pruning.
* **Stateless sessions** — resume locates the newest `last.pt` under
  `/kaggle/input` and continues from the stored epoch/step; finished cells are
  skipped via `results/metrics/ablation_manifest.json`.
* **Portable inputs** — the dataset and codebase arrive as zips with forward
  slashes (made by `scripts/make_zips.py`, which verifies every entry).
