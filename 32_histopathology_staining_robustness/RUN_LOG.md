# RUN_LOG — Ablation Run Register

One section per cell of the ablation matrix (`PLAN.md` §3). Fill in one **attempt
block per Kaggle session** used on that cell — a cell that spans two sessions has
two blocks, which is the normal case given the 10.5 h time box.

**How to fill this in.** After each session, download the version's `results/`
folder. Most fields are copy-paste from:

| Field | Where to read it |
| :--- | :--- |
| Session / notebook version | Kaggle → notebook → **Versions** list (e.g. `v7`) |
| Date | the version's creation date |
| Config / hash | `results/per_experiment/<EXP>/robustness.json` → `config_hash` |
| Status / stop reason / epochs | `checkpoints/<EXP>/train_report.json` → `train_result.status`, `.stop_reason`, `.epochs_completed` |
| Val Macro-F1 / best epoch | same file → `train_result.best_val_macro_f1`, `.best_epoch` |
| Metrics | `results/per_experiment/<EXP>/evaluation_result.txt` |
| Wall time | `train_result.trained_seconds` / 60, or `results/metrics/session_summary.json` |
| Checkpoint size | `checkpoints/<EXP>/checkpoint_report.json` → `current_gb` |

Anything not yet run stays `pending`. Do not pre-fill numbers for runs that have
not happened.

Status vocabulary: `pending` · `running` · `completed` · `interrupted` (time box
fired; will resume) · `failed` · `rerun` (superseded by a later attempt block).

Global identifiers for this project:

* Notebook: `<paste Kaggle notebook URL here>`
* Codebase dataset: `histo-robust-code` (version used per block)
* Checkpoint dataset: `histo-robust-checkpoints`
* Global seed: `42` (split seed and runtime seed are both 42 by design)

---

## EXP-01 — Stage 0 — Primary Baseline

**ResNet-50 (ImageNet-1k) · normalization: none · augmentation: none**
*Research question:* Out-of-the-box in-domain and OOD performance under raw
staining. Comparator: anchor for every other cell.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp01_baseline_resnet50.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=resnet50` · `model.pretrained=true` · `model.freeze_backbone=false` · `normalization.name=none` · `augmentation.policy=none` · `data.batch_size=64` · `train.epochs=12` · `train.lr=0.001` · `train.weight_decay=0.05` · `train.grad_accum_steps=1` · `train.amp=true` · `train.scheduler=cosine` · `checkpoints.save_every_steps=2000` · `checkpoints.keep_last=3` |
| Command | `python scripts/train.py --config configs/experiments/exp01_baseline_resnet50.yaml --exp-id EXP-01 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | pending |

---

## EXP-02 — Stage 1 — Normalization: Reinhard

**ResNet-50 · normalization: Reinhard (CIELAB) · augmentation: none**
*Research question:* Does statistical Lab colour transfer improve out-of-domain
generalisation? Comparator: EXP-01.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp02_norm_reinhard_resnet50.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=resnet50` · `normalization.name=reinhard` · `normalization.reference_path=data/processed/templates/reference_stain.png` · `normalization.params.use_tissue_mask=true` · `normalization.params.luminance_threshold=220` · `augmentation.policy=none` · `data.batch_size=64` · `train.epochs=12` · `train.lr=0.001` |
| Reference tile provenance | pending (record `reference_provenance` from `checkpoints/EXP-02/train_report.json`) |
| Command | `python scripts/train.py --config configs/experiments/exp02_norm_reinhard_resnet50.yaml --exp-id EXP-02 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | pending |

---

## EXP-03 — Stage 1 — Normalization: Macenko

**ResNet-50 · normalization: Macenko (OD H&E deconvolution) · augmentation: none**
*Research question:* Does optical-density H&E deconvolution outperform statistical
colour transfer? Comparator: EXP-01, EXP-02.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp03_norm_macenko_resnet50.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=resnet50` · `normalization.name=macenko` · `normalization.params.od_threshold=0.15` · `normalization.params.angular_percentile=99.0` · `normalization.params.concentration_percentile=99.0` · `normalization.params.alpha=1.0` · `normalization.params.background_fraction_limit=0.85` · `augmentation.policy=none` · `data.batch_size=64` · `train.epochs=12` |
| Reference tile provenance | pending |
| Macenko fallback counters | pending (`normalizer_background_skips`, `normalizer_exceptions` from `results/per_experiment/EXP-03/robustness.json` or the normalisation cache manifest) |
| Command | `python scripts/train.py --config configs/experiments/exp03_norm_macenko_resnet50.yaml --exp-id EXP-03 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | pending |

---

## EXP-04 — Stage 2 — Augmentation: Geometric only

**ResNet-50 · normalization: none · augmentation: Aug-Geo (flips + rot90)**
*Research question:* What portion of robustness is attributable solely to spatial
invariance? Comparator: EXP-01.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp04_aug_geo_resnet50.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=resnet50` · `normalization.name=none` · `augmentation.policy=aug_geo` · `augmentation.geometric.hflip_prob=0.5` · `vflip_prob=0.5` · `rot90_prob=0.5` · `data.batch_size=64` · `train.epochs=12` |
| Command | `python scripts/train.py --config configs/experiments/exp04_aug_geo_resnet50.yaml --exp-id EXP-04 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | pending |

---

## EXP-05 — Stage 2 — Augmentation: Stain (HED jitter)

**ResNet-50 · normalization: none · augmentation: Aug-Stain**
*Research question:* Does synthetic HED stain jitter match or exceed explicit
stain normalisation? Comparator: EXP-01, EXP-03.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp05_aug_stain_resnet50.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=resnet50` · `normalization.name=none` · `augmentation.policy=aug_stain` · `augmentation.stain.he_scale=[0.85,1.15]` · `he_shift=[-0.08,0.08]` · `brightness=0.10` · `contrast=0.10` · `hue=0.03` · `saturation=0.12` · `apply_prob=0.8` · `data.batch_size=64` · `train.epochs=12` |
| Command | `python scripts/train.py --config configs/experiments/exp05_aug_stain_resnet50.yaml --exp-id EXP-05 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | pending |

---

## EXP-06 — Stage 2 — Augmentation: Combined

**ResNet-50 · normalization: none · augmentation: Aug-Combined (geo + stain)**
*Research question:* Are spatial and stain augmentations additive when used
without normalisation? Comparator: EXP-04, EXP-05.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp06_aug_combined_resnet50.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=resnet50` · `normalization.name=none` · `augmentation.policy=aug_combined` · geometric + stain params as EXP-04/EXP-05 · `data.batch_size=64` · `train.epochs=12` |
| Command | `python scripts/train.py --config configs/experiments/exp06_aug_combined_resnet50.yaml --exp-id EXP-06 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | pending |

---

## EXP-07 — Stage 3 — Interaction: Macenko × Aug-Geo

**ResNet-50 · normalization: Macenko · augmentation: Aug-Geo**
*Research question:* Does combining Macenko normalization with spatial
augmentation yield gains? Comparator: EXP-03, EXP-04.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp07_interaction_macenko_geo_resnet50.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=resnet50` · `normalization.name=macenko` (params as EXP-03) · `augmentation.policy=aug_geo` · `data.batch_size=64` · `train.epochs=12` |
| Command | `python scripts/train.py --config configs/experiments/exp07_interaction_macenko_geo_resnet50.yaml --exp-id EXP-07 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | pending |

---

## EXP-08 — Stage 3 — Interaction: Macenko × Aug-Stain

**ResNet-50 · normalization: Macenko · augmentation: Aug-Stain**
*Research question:* Does stain jitter add value to normalized tiles, or do they
conflict? Comparator: EXP-03, EXP-05.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp08_interaction_macenko_stain_resnet50.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=resnet50` · `normalization.name=macenko` · `augmentation.policy=aug_stain` (params as EXP-05) · `data.batch_size=64` · `train.epochs=12` |
| Command | `python scripts/train.py --config configs/experiments/exp08_interaction_macenko_stain_resnet50.yaml --exp-id EXP-08 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | pending |

---

## EXP-09 — Stage 3 — Defended ResNet Baseline

**ResNet-50 · normalization: Macenko · augmentation: Aug-Combined**
*Research question:* Peak ResNet performance under the full defence.
Comparator: EXP-01, EXP-06.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp09_interaction_macenko_combined_resnet50.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=resnet50` · `normalization.name=macenko` · `augmentation.policy=aug_combined` · `data.batch_size=64` · `train.epochs=12` |
| Command | `python scripts/train.py --config configs/experiments/exp09_interaction_macenko_combined_resnet50.yaml --exp-id EXP-09 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | pending |

---

## EXP-10 — Stage 4 — Backbone: ConvNeXt-Tiny (raw)

**ConvNeXt-Tiny (ImageNet-1k) · normalization: none · augmentation: none**
*Research question:* Is a modern ConvNet intrinsically more robust to stain shift
than ResNet? Comparator: EXP-01.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp10_backbone_convnext_raw.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=convnext_tiny` · `model.pretrained=true` · `normalization.name=none` · `augmentation.policy=none` · `data.batch_size=64` · `train.epochs=12` |
| Command | `python scripts/train.py --config configs/experiments/exp10_backbone_convnext_raw.yaml --exp-id EXP-10 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | pending |

---

## EXP-11 — Stage 4 — Backbone: ConvNeXt-Tiny (defended)

**ConvNeXt-Tiny · normalization: Macenko · augmentation: Aug-Combined**
*Research question:* Does ConvNeXt-Tiny benefit from the combined
normalization/augmentation policy? Comparator: EXP-09, EXP-10.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp11_backbone_convnext_best.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=convnext_tiny` · `normalization.name=macenko` · `augmentation.policy=aug_combined` · `data.batch_size=64` · `train.epochs=12` |
| Command | `python scripts/train.py --config configs/experiments/exp11_backbone_convnext_best.yaml --exp-id EXP-11 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | pending |

---

## EXP-12 — Stage 4 — Backbone: Phikon (raw)

**Phikon ViT-B/16 (TCGA iBOT, frozen / linear probe) · normalization: none · augmentation: none**
*Research question:* Does large-scale histology SSL pretraining provide intrinsic
stain invariance without defences? Comparator: EXP-01, EXP-10.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp12_backbone_phikon_raw.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=phikon` · `model.pretrained=true` · `model.freeze_backbone=true` · `model.dropout=0.0` · `normalization.name=none` · `augmentation.policy=none` · `data.batch_size=32` · `train.grad_accum_steps=2` (effective batch 64) · `train.epochs=12` |
| Weights source | `owkin/phikon` via HF `transformers` (ungated). Record whether Internet was ON or the weights came from a mounted dataset. |
| Command | `python scripts/train.py --config configs/experiments/exp12_backbone_phikon_raw.yaml --exp-id EXP-12 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | `freeze_backbone=true` trains the head on cached features, which is far cheaper than full fine-tuning. If it is later decided to fine-tune Phikon, record that as a deviation and note the resulting VRAM/batch change. |

---

## EXP-13 — Stage 4 — Backbone: Phikon (defended)

**Phikon ViT-B/16 (frozen / linear probe) · normalization: Macenko · augmentation: Aug-Combined**
*Research question:* Can foundation-model representations be boosted further by
stain normalization and augmentation? Comparator: EXP-09, EXP-11, EXP-12.

### Attempt 1

| Field | Value |
| :--- | :--- |
| Kaggle notebook link / version | pending |
| Date run | pending |
| Session number | pending |
| Config file | `configs/experiments/exp13_backbone_phikon_best.yaml` |
| Effective config hash | pending |
| Exact config (key fields) | `model.backbone=phikon` · `model.freeze_backbone=true` · `model.dropout=0.0` · `normalization.name=macenko` · `augmentation.policy=aug_combined` · `data.batch_size=32` · `train.grad_accum_steps=2` · `train.epochs=12` |
| Weights source | `owkin/phikon` via HF `transformers` (ungated) |
| Command | `python scripts/train.py --config configs/experiments/exp13_backbone_phikon_best.yaml --exp-id EXP-13 --output-dir /kaggle/working/checkpoints --eval-test-id --eval-test-ood` |
| Status | pending |
| Stop reason | pending |
| Epochs completed / planned | pending / 12 |
| Wall time (min) | pending |
| Best val Macro-F1 (epoch) | pending |
| test_id Macro-F1 / test_ood Macro-F1 | pending / pending |
| delta_f1 / rr_f1 | pending / pending |
| Checkpoint size (GB) | pending |
| Deviations from plan | none |
| Notes | pending |

---

## Session Index

One row per Kaggle session. This is the fastest way to see how the matrix was
spread across the 12-hour budget.

| Session | Notebook version | Date (UTC) | Cells attempted | Cells completed | Session trained (min) | Stop reason | Outputs downloaded? | Notes |
| :---: | :---: | :--- | :--- | :--- | ---: | :--- | :---: | :--- |
| 1 | pending | pending | pending | pending | pending | pending | pending | pending |
| 2 | pending | pending | pending | pending | pending | pending | pending | pending |
| 3 | pending | pending | pending | pending | pending | pending | pending | pending |

---

## Infrastructure notes

Record environment facts once per session here rather than repeating them per
cell — they are what a reader needs to judge reproducibility.

| Session | GPU | torch / CUDA | Internet ON? | Codebase dataset version | Checkpoint dataset version | Pre-flight verdict | Notes |
| :---: | :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| 1 | pending | pending | pending | pending | n/a | pending | pending |
| 2 | pending | pending | pending | pending | pending | pending | pending |

---

## Known open items (fill in as they arise)

| # | Item | Status | Resolution |
| :-: | :--- | :--- | :--- |
| 1 | Canonical stain reference tile is auto-selected from the training split (`prepare_splits.py --reference-out`). | resolved by design | Deterministic TUM tile; provenance recorded in `train_report.json` → `reference_provenance` and `split_summary.json` → `reference_source_image`. |
| 2 | Compute budget: 25,000-patch stratified subset vs full 100,000. | pending | Default is the 25k subset; record here if `FULL = True` was used, since it changes the epoch count that fits in a session. |
| 3 | Normalisation cache used or not. | pending | Record per session: online normalisation vs `--normalization-cache` (cached runs disable the online normaliser and are equivalent by construction — verify via `config_hash`). |
