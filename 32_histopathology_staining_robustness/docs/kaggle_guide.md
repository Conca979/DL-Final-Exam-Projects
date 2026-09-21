# Kaggle Setup Guide — Robust Histopathology Classification under Staining Variations

> `docs/` index: [`README.md`](README.md) · plan: [`PLAN.md`](PLAN.md) · dataset:
> [`dataset_card.md`](dataset_card.md) · run register: [`RUN_LOG.md`](RUN_LOG.md) ·
> results: [`RESULTS.md`](RESULTS.md) · reference: [`APPENDICES.md`](APPENDICES.md)

This is the exact, click-by-click procedure for running the 13-cell ablation matrix
(`docs/PLAN.md` §3) on Kaggle's free GPU tier.

Read the four rules first — they are the ones that actually cost people runs:

| Rule | Consequence if ignored |
| :--- | :--- |
| **Zip with forward slashes, never the Windows "Send to → Compressed folder" tool.** | Kaggle extracts one flat file literally named `data\raw\ADI\a.png` instead of a directory tree. |
| **Never let `/kaggle/working` exceed 20 GB.** | Kaggle kills the session's outputs; checkpoints and metrics are lost with no warning. |
| **Never plan to run past 12 h.** | Kaggle terminates the session at exactly 12:00:00 with no cleanup hook. |
| **Use "New Version" on the checkpoint dataset, never a new dataset.** | You accumulate a dozen near-identical datasets and the notebook has to be re-pointed every session. |

The code handles the last three automatically. The first one is on you, and it is
mechanical — use the provided script.

---

## 0. What you need before you start

* A Kaggle account with phone verification (free GPU requires it).
* The two dataset archives from Zenodo record [1214456](https://doi.org/10.5281/zenodo.1214456):
  * `NCT-CRC-HE-100K-NONORM.zip` (~1.4 GB compressed, 100,000 patches)
  * `CRC-VAL-HE-7K.zip` (~105 MB compressed, 7,180 patches)
* This repository on disk.

No approval walls, no credentials, no registration — both files are direct HTTP
downloads under CC-BY 4.0 (`docs/dataset_card.md` §3).

---

## 1. The Zipping Rule (do this first, it prevents the #1 failure)

### 1a. Why the Windows zip button breaks Kaggle

Windows' built-in "Compressed (zipped) folder" writes archive entry names with
**backslashes**: `data\raw\ADI\patch.png`. Linux — and therefore Kaggle — treats
a backslash as an ordinary filename character, so the extraction produces a
single file whose *name* contains backslashes rather than nested directories.
You then spend an hour debugging a "missing class folder" error that has nothing
to do with your data.

`scripts/make_zips.py` never has this problem: Python's `zipfile` always writes
`/`-separated names, and the script re-opens every archive it produces and
verifies that no entry contains a backslash, an absolute path or a drive letter.

### 1b. Run it

From the repository root:

```powershell
python scripts/make_zips.py --out-dir dist
```

Expected output (sizes vary slightly):

```text
[1/3] codebase
  wrote histo-robust-code.zip: 54 files, 0.15 MB
[2/3] datasets
  copied NCT-CRC-HE-100K-NONORM.zip (11183 MB)
  copied CRC-VAL-HE-7K.zip (763 MB)
[3/3] checkpoints
  skipped (pass --checkpoints <dir> to bundle a previous session)

[OK  ] histo-robust-code.zip: 54 entries
[OK  ] NCT-CRC-HE-100K-NONORM.zip: 100010 entries
[OK  ] CRC-VAL-HE-7K.zip: 7190 entries
```

`dist/` now holds the three upload artifacts. The two dataset archives are
copied verbatim from `zipped_dataset/` (Zenodo's own archives already use
forward slashes — verified above). Add `--repack-data` only if you have the raw
PNG folders and no archive; it re-creates them from scratch and takes a long
time for the 100K set.

> Later, when a session ends and you want to carry the checkpoints forward:
> `python scripts/make_zips.py --out-dir dist --checkpoints checkpoints --zip-name-version v2`
> produces `histo-robust-checkpoints-v2.zip` with a `checkpoints/` prefix so that
> mounting it at `/kaggle/input` reproduces the expected layout.

### 1c. Manual alternative (Linux / WSL only)

If you would rather do it by hand, use `zip` and never a GUI tool:

```bash
cd /path/to/32_histopathology_staining_robustness
zip -r dist/histo-robust-code.zip \
    src scripts configs kaggle tests docs \
    pyproject.toml README.md \
    -x '*/__pycache__/*' '*.pyc' 'data/*' 'checkpoints/*' 'results/*' 'dist/*'
```

Verify before uploading:

```bash
python -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); bad=[n for n in z.namelist() if '\\\\' in n]; print('BAD' if bad else 'OK', len(z.namelist()), bad[:3])" dist/histo-robust-code.zip
```

---

## 2. Create the three (then four) Kaggle datasets

Go to <https://www.kaggle.com/datasets> → **New Dataset** for each item below.
All four are **private** — this project's results are not public.

| # | Dataset title (use exactly this slug) | Upload file | Notes |
| :-: | :--- | :--- | :--- |
| 1 | `histo-robust-code` | `dist/histo-robust-code.zip` | The codebase. Re-upload a **New Version** whenever you change any `.py`/`.yaml`. |
| 2 | `nct-crc-he-100k-nonorm` | `dist/NCT-CRC-HE-100K-NONORM.zip` | In-domain source. ~11 GB once extracted; uploading the zip is ~1.4 GB. |
| 3 | `crc-val-he-7k` | `dist/CRC-VAL-HE-7K.zip` | Out-of-domain target. |
| 4 | `histo-robust-checkpoints` | *created after session 1* | See §5. Do **not** create it now with a placeholder. |

Uploading tips that matter at these sizes:

* Use the Kaggle **web uploader** and leave the browser tab open until it says
  "Completed". The 1.4 GB source archive typically takes 10–30 minutes on a
  domestic uplink.
* Do **not** unzip before uploading. Kaggle's web uploader handles a zip file
  fine and chokes on a folder containing 100,000 PNGs.
* Set visibility to **Private**.

After creating datasets 2 and 3, the notebook finds them automatically (it
searches `/kaggle/input` by name *and* by content — it looks for directories that
contain at least six of the nine class folders), so the slugs do not have to match
exactly.

---

## 3. Create the notebook and attach everything

1. **Code** → **New Notebook**.
2. **File → Import Notebook** → upload
   `kaggle/notebook_01_run_all_ablations.ipynb` from this repository.
3. Open the right-hand **Settings** panel and set:
   * **Accelerator**: `GPU P100` (preferred) or `GPU T4 x2`.
   * **Internet**: `On` for the first session — pretrained weights are downloaded
     and then cached in `~/.cache` for the rest of the session.
   * **Persistence**: `Files only` (there is nothing to persist; state travels in
     the checkpoint dataset).
4. Open the **Data** panel → **Add Input** → **Your Datasets** and attach:
   * `histo-robust-code`
   * `nct-crc-he-100k-nonorm`
   * `crc-val-he-7k`
   * (from session 2 onwards) `histo-robust-checkpoints`
5. Verify the notebook is *interactive* before committing to a 10-hour run:
   run cells 1 → 2 → 3 → 4 and read the pre-flight verdict.

### What each cell does

| Cell | Purpose | Typical runtime |
| :--- | :--- | :--- |
| **1 — Setup** | Finds the codebase zip anywhere under `/kaggle/input`, extracts it, installs what is missing (`pip install -e .`), **hard-asserts CUDA is present**, and reports free disk. | 1–3 min |
| **2 — Data mount** | Locates both datasets (name- and content-based), points training straight at the read-only `/kaggle/input` mount so ~12 GB of raw images never consume the 20 GB output quota, and copies any previous session's checkpoints into `/kaggle/working/checkpoints` so resume works. | 1–2 min |
| **3 — Splits + reference** | Runs `scripts/prepare_splits.py`: stratified 70/15/15 source split at seed 42, the full target set as `test_ood`, and one canonical H&E reference tile chosen **from the training split only**. Asserts the domain firewall. | 2–5 min |
| **4 — Pre-flight benchmark** | `scripts/smoke_test.py`: measures DataLoader throughput and real forward/backward step time, extrapolates minutes/epoch, and prints `OK` / `CPU-BOUND` / `IDLE-GPU` with advice. | 3–8 min |
| **4b — (optional) cache** | Builds the offline normalisation cache. Only needed if cell 4 says `CPU-BOUND`, or to fit more cells per session. | 20–60 min |
| **5 — The run** | `scripts/run_all_ablations.py`: trains every pending cell under the 10.5 h time box, evaluates in-domain and out-of-domain, and appends to the manifest. | up to ~11 h |
| **6 — Artifacts** | Zips the checkpoints into `for_upload/` and prints the download list. | 1–3 min |

### Reading the pre-flight verdict

```text
  loader throughput   : 240.5 img/s (0.27 s/batch)
  train step          : 0.19 s (337 img/s)
  peak GPU memory     : 3.9 GB
  minutes per epoch   : 1.8
  epochs in budget    : 6.9 of 12 planned
  VERDICT             : OK
```

* `OK` — go ahead.
* `CPU-BOUND` (loader slower than the GPU step) — enable cell 4b and set
  `NORMALIZATION_CACHE` in cell 5.
* `IDLE-GPU` — raise `data.num_workers` (4 is the Kaggle vCPU count; 2 is
  sometimes faster because of memory pressure) or enable cell 4b.
* `epochs in budget` below ~3 — switch `SUBSET = 25000` (cell 3) to the default
  or set `freeze_backbone: true` in the config for the Phikon cells.

---

## 4. Execution: trigger the background run

1. **Save Version** (top right) → choose **Save & Run All (Commit)** → **Save**.
2. Kaggle immediately detaches the run. You can close the browser; the run
   continues in the background.
3. The session is hard-killed at **12 h**. The code stops itself at **630 min
   (10.5 h)** by default and also reserves **20 min** before the session
   deadline — whichever comes first — and before exiting it:
   * breaks at a batch boundary (never mid-write),
   * writes `last.pt` (model + optimizer + scheduler + AMP scaler + RNG state) and
     `best.pt` (best `val_id` macro-F1),
   * evaluates the frozen checkpoint on `test_id` and `test_ood`,
   * prunes checkpoints and updates the manifest.

   That is why the run ends "successfully" rather than as a 12-hour timeout: a
   clean exit is what makes the next session resumable.

### Monitoring

`results/logs/run_all_ablations.log` is the full transcript. Watch for:

```text
EXP-03 | epoch 4/12 | train_loss=0.4213 | val_acc=0.9102 val_macro_f1=0.8931 ... | 38.4 min elapsed
EXP-03 | TIME BOX REACHED (run_time_budget) after epoch 7. Saving last.pt and exiting cleanly ...
```

`[TimeBudget:EXP-03] elapsed=75.0 min | remaining=0.0 min | session_left=214.6 min` is
printed at the top of every epoch and is the number to watch if you are unsure
how much of the session is left.

---

## 5. Updating checkpoints between sessions (**the important bit**)

A Kaggle session is stateless: `/kaggle/working` is wiped when the version
finishes. The only way a 10.5-hour run survives is if you move its checkpoints
back into `/kaggle/input` as a **new version of the existing dataset**.

Do this every time a session ends:

1. **Download the outputs immediately.** Open the finished version → **Output**
   tab → download:
   * `for_upload/histo-robust-checkpoints.zip` ← the resumable state
   * the whole `results/` folder ← metrics, confusion matrices, logs

   Kaggle keeps outputs for a while, but downloading promptly is the only way to
   be certain. **Do not** skip this step and assume you can fetch it later.

2. **Update the dataset, do not create a new one.**
   * Go to the `histo-robust-checkpoints` dataset page.
   * Click **New Version** (top right).
   * Upload the zip you just downloaded over the previous file.
   * **Save** — do *not* rename the dataset and do *not* leave the old file in
     place. A new version keeps the same mount path in every notebook that
     already references it, so the notebook needs zero edits.
   * On the very first cycle the dataset does not exist yet: create it once with
     `histo-robust-checkpoints` as the title and the zip as the file, then use
     **New Version** forever after.

3. **Re-run the same notebook version.**
   * Attach the freshly updated `histo-robust-checkpoints` dataset (it stays
     attached across runs of the same notebook).
   * **Save Version → Save & Run All (Commit)** again.

   Cell 2 copies every `*.pt` it finds under `/kaggle/input` into
   `/kaggle/working/checkpoints`; cell 5 then:
   * **skips** every cell already marked `completed` in
     `results/metrics/ablation_manifest.json`, and
   * **resumes** the interrupted cell from `last.pt`, continuing at the stored
     epoch and global step with the optimizer/scheduler/RNG state restored.

   With the default per-cell cap (75 min) and a 700-minute session, expect
   roughly 7–9 cells per session, i.e. **the full matrix in two sessions**.

### Why the manifest matters

If you lose `results/metrics/ablation_manifest.json` (for example you only
downloaded the checkpoint zip), the orchestrator can no longer tell which cells
are finished and will re-run them. Two guards:

* The zip from `scripts/make_zips.py --checkpoints` is a *bundle*, not a
  replacement: keep downloading `results/` as well, or
* pass `--force` to deliberately re-run a cell whose checkpoint you still have —
  resuming from `last.pt` costs only the remaining epochs.

---

## 6. Resuming, re-running, and partial runs

| Goal | How |
| :--- | :--- |
| Resume everything after a session ended | Just re-run the notebook. Nothing else. |
| Run only two specific cells | In cell 5: `EXPERIMENTS = ["EXP-01", "EXP-03"]` |
| Re-run a cell from scratch | Delete `checkpoints/EXP-xx/` in cell 2 (or before cell 5), then run. |
| Re-run a cell but keep its progress | Leave the folder in place; it resumes from `last.pt`. |
| Re-evaluate without training | `python scripts/evaluate.py --config configs/experiments/exp03_...yaml --exp-id EXP-03 --checkpoint-dir /kaggle/working/checkpoints` |
| See the allocation before committing | Set `DRY_RUN = True` in cell 5. |
| Train on all 100,000 patches | Cell 3: `FULL = True` (no code changes; expect many more sessions). |
| Free GPU quota exhausted | The notebook is resumable by construction — stop and continue tomorrow. |

---

## 7. Offline model weights (Internet = OFF)

Only the Phikon cells (`EXP-12`, `EXP-13`) and the timm ImageNet weights need a
download. With Internet ON they are fetched once and cached for the session. If
you must run with Internet OFF, mount the weights as a dataset:

1. On a machine with Internet, download both:
   * timm: `python -c "import timm; timm.create_model('resnet50.a1_in1k', pretrained=True); timm.create_model('convnext_tiny.fb_in1k', pretrained=True)"`
     → cached under `~/.cache/huggingface/hub/`
   * Phikon: `huggingface-cli download owkin/phikon --local-dir phikon`
     (Phikon is **ungated**, so no token is required.)
2. Zip the cache with forward slashes and upload it as a dataset, e.g.
   `histo-robust-weights`, so it mounts at
   `/kaggle/input/histo-robust-weights/`.
3. In cell 5, or via the config, point the loader at it:

   ```python
   command += ["--weights-dir", "/kaggle/input/histo-robust-weights"]
   ```
   or in a config file: `paths: { weights_dir: /kaggle/input/histo-robust-weights }`

The loader recognises `phikon/`, `models--owkin--phikon/snapshots/<rev>/` and a
plain directory containing `config.json` + `model.safetensors`.

---

## 8. Storage budget (the 20 GB cap, by the numbers)

| Item | Size | Notes |
| :--- | ---: | :--- |
| Codebase in `/kaggle/working` | ~1 MB | |
| Raw images | **0 GB** | Read directly from the read-only `/kaggle/input` mount. |
| Splits CSV + reference tile | < 5 MB | |
| Optional Macenko cache (all four splits) | ~3 GB | Only if you run cell 4b. |
| Checkpoints, ResNet-50/ConvNeXt | ~2.5 GB | 50 MB × (3 rolling + 3 best) × 13 cells with pruning active. |
| Checkpoints, Phikon (linear probe) | ~1.5 GB | Head-only checkpoints are small. |
| Results (metrics, predictions, figures) | ~200 MB | |
| Checkpoint zip in `for_upload/` | ~2.5 GB | Needed for the next session. |
| **Total, typical** | **~9 GB** | Comfortably inside 20 GB. |

Guard rails already in the code:

* `checkpoints.save_every_steps` (`--save-every`) controls rolling-checkpoint
  granularity — default 2000 steps.
* `checkpoints.keep_last` (`--keep-last 3`) keeps only the newest three
  `step_*.pt` per cell.
* `checkpoints.keep_best` (3) keeps the three best `best_v*.pt` snapshots;
  `best.pt` and `last.pt` are always retained because resume needs them.
* `checkpoints.run_quota_gb` (3.0) is a per-cell ceiling: the manager prunes the
  oldest rolling checkpoints until the cell fits, and logs what it removed.
* Cell 1 refuses to continue if less than 6 GB is free.

If you are tight on space, lower `keep_last` to `1` in cell 5
(`command += ["--keep-last", "1"]`).

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `FATAL: could not find the codebase zip` | Dataset 1 not attached, or it contains the folder rather than the zip | Attach `histo-robust-code`; re-create it from `histo-robust-code.zip`. |
| `FATAL: could not locate the datasets` / `missing class folders` | Dataset uploaded unzipped-but-flattened, or extracted with backslashes | Re-upload the original Zenodo zip (or a Python-made zip). |
| `No CUDA device is available` | Accelerator not set, or set after the kernel started | Settings → Accelerator → `GPU P100`; then re-run cell 1. |
| `CUDA out of memory` | batch size too high for the card | The orchestrator retries automatically with a halved batch and doubled accumulation (`--oom-retries 2`). To force it: `command += ["--batch-size", "32"]`. |
| `HTTPError` / `Couldn't reach huggingface.co` during cell 5 | Internet is OFF or the download was throttled | Enable Internet for the first run, or mount the weights (§7). Only `EXP-12`/`EXP-13` actually need Phikon. |
| Run ended after ~10.5 h with status "interrupted" | This is the design: `run_time_budget` fired before the 12 h kill. | Not a failure. Download the outputs and continue with a new version (§5). |
| `results/metrics/summary_results.csv` missing some cells | Those cells had not finished when the session ended | Expected in multi-session runs; the manifest records exactly what remains. |
| Val macro-F1 much lower than ~0.9 | Only a few epochs ran (time-boxed), or the LR/batch was overridden | Check `epochs in budget` from cell 4 and the per-cell budget in the log. |
| Everything looks fine but the manifest says `failed` for Phikon | Missing `transformers` or no network | The error is recorded verbatim in the manifest and the run log. |

---

## 10. Reproducing a single cell locally (non-Kaggle)

The scripts are plain CLI tools; nothing about them is Kaggle-specific except the
default paths.

```bash
pip install -e .
python scripts/prepare_splits.py --data-root data/raw/NCT-CRC-HE-100K-NONORM \
    --target-root data/raw/CRC-VAL-HE-7K --out-dir data/processed/splits \
    --subset 25000 --seed 42 --reference-out data/processed/templates/reference_stain.png

python scripts/smoke_test.py --config configs/experiments/exp01_baseline_resnet50.yaml

python scripts/train.py --config configs/experiments/exp01_baseline_resnet50.yaml \
    --exp-id EXP-01 --output-dir checkpoints --results-dir results \
    --max-minutes 630 --eval-test-id --eval-test-ood

python scripts/run_all_ablations.py --checkpoint-dir checkpoints --results-dir results \
    --per-exp-minutes 75 --eval-test-ood
```

`python tests/local_selftest.py` runs 50+ dependency-light checks over the
normalisation, augmentation, metric, time-budget and checkpoint-pruning code —
useful as a 30-second sanity check before spending GPU hours.
