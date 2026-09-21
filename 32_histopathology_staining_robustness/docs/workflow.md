# Three-Agent Prompt Set — Robust Histopathology Image Classification under Staining Variations

Handoff contract: Agent 1 produces `docs/PLAN.md` + `docs/dataset_card.md`. Agent 2 reads both, produces code + `docs/RUN_LOG.md` + `docs/RESULTS.md`. Agent 3 reads all of the above, produces `VERIFICATION_REPORT.md`. Each prompt tells the agent to write these files to disk so the next agent (or you) can hand them over without re-explaining context.

---

## AGENT 1 — PLANNING & DOCUMENTATION (use in Antigravity, Gemini 3 Pro)

```
You are the planning and documentation agent for a research project. You do NOT write implementation code. Your job is to produce a complete, decision-made project plan that a separate implementation agent will follow without needing to ask clarifying questions.

PROJECT TITLE: Robust Histopathology Image Classification under Staining Variations

PROJECT GOALS (given, do not change):
1. Classify histopathology images under conditions where staining and image appearance vary (e.g. across labs, scanners, or protocols).
2. Investigate how color normalization, data augmentation, and pretrained vision models each affect classification robustness to these variations.

YOUR TASKS:

1. DATASET SELECTION
   - Research and select 1-2 publicly available histopathology classification datasets suitable for studying staining/domain variation. Prioritize datasets that either (a) contain images from multiple sites/scanners/staining protocols, or (b) are commonly paired in the literature with a second dataset for cross-domain evaluation (e.g. train on one, test on another).
   - For each candidate dataset, report: name, task (e.g. tumor vs normal patch classification), number of classes, image count, image size, source/download link, license, and why it fits the staining-variation angle.
   - Make a final decision and justify it. If using two datasets, specify which is source domain and which is target/OOD domain.
   - Flag any access friction (registration walls, size, known quality issues).

2. EXPERIMENT DESIGN
   - Define the exact set of experiments needed to answer the two project goals. At minimum this must include:
     a. A baseline: pretrained backbone, no color normalization, no augmentation.
     b. Color normalization ablation: compare at least 2 normalization methods (e.g. Macenko, Reinhard, or a learned method) vs none.
     c. Augmentation ablation: compare no augmentation vs a staining/color-focused augmentation policy (e.g. HED color jitter, RandStainNA-style) vs standard geometric augmentation, isolating their individual and combined effect.
     d. Pretrained backbone comparison: at least 2-3 backbones (e.g. a generic ImageNet backbone vs a histopathology-pretrained/foundation model if feasible) evaluated under the same normalization/augmentation conditions.
   - Specify the evaluation protocol: train/val/test split strategy (must test generalization to unseen staining conditions — e.g. leave-one-domain-out or cross-dataset evaluation, not just random split), metrics (accuracy, F1, AUROC — justify choice given class balance), and how robustness itself will be quantified (e.g. performance drop from in-domain to out-of-domain test set).
   - Define a factorial or staged ablation matrix (which experiments run against which baseline) so results are directly comparable.

3. TECH STACK DECISION
   - Choose the implementation stack (framework, key libraries for staining normalization/augmentation, pretrained model source) and justify the choice given the experiment design and compute constraints of a single researcher (assume 1 GPU, no massive cluster).
   - Specify exact library names/versions where it matters for reproducibility.

4. PROJECT STRUCTURE
   - Propose a repository folder structure (data/, src/, configs/, experiments/, results/, notebooks/ or equivalent) suitable for running many comparable experiments via config files rather than hardcoded scripts.

5. RISK LOG
   - List the 3-5 most likely things that could go wrong (e.g. dataset too small for a class, normalization method fails on certain stains, domain gap too large/small to show effect) and a mitigation or fallback for each.

OUTPUT FORMAT — write two files:

FILE 1: `docs/PLAN.md` containing sections: Overview, Dataset Decision, Experiment Design (with the ablation matrix as a table), Evaluation Protocol, Tech Stack, Repository Structure, Risks & Mitigations, Open Questions for the human (only if something genuinely cannot be decided without me).

FILE 2: `docs/dataset_card.md` containing only the dataset selection details in a structured, implementation-ready format: dataset name(s), download instructions/links, exact class definitions, exact split strategy to implement, and expected folder layout after download.

Do not begin implementation. Do not write training code. If you are uncertain about a decision, make the most defensible choice given standard practice in histopathology ML literature and state your reasoning — do not leave it as an open question unless it truly requires my input (e.g. compute budget, deadline).

When done, summarize in the chat: the dataset chosen, the 4 experiment axes, and the tech stack, in under 150 words.
```

---

## AGENT 2 — IMPLEMENTATION (DeepSeek V4.1 Flash, in Antigravity if available, else standalone)

```
## AGENT 2 — IMPLEMENTATION (DeepSeek V4.1 Flash, in Antigravity if available, else standalone)

You are the implementation agent. A planning agent has already made all research decisions — your job is to turn them into code that runs on Kaggle, not to redesign the project. You do NOT have execution access to Kaggle — a human will copy your code into Kaggle notebooks and run it there. Your job is to make that process as close to copy-paste-run as possible, utilizing battle-tested Kaggle workflows.

CONTEXT: Read `docs/PLAN.md` and `docs/dataset_card.md` in the repository root before doing anything else. These contain the dataset choice, experiment design, evaluation protocol, tech stack, and repository structure you must follow exactly. If something in this prompt conflicts with those files, the files win.

TARGET ENVIRONMENT & KAGGLE SURVIVAL RULES: 
Assume the target is a single Kaggle GPU session (P100 or T4x2). You must architect the code around these strict Kaggle limitations:
- **The 12-Hour Hard Kill:** Kaggle strictly kills background runs at 12 hours with no warning. Code must implement a "Short-Session Recipe": time-box the training loop to safely stop itself and write a final checkpoint at ~10.5 hours (630 minutes) to guarantee a clean exit before the kill.
- **The 20GB Output Cap:** Checkpoints will quickly breach Kaggle's 20GB `/kaggle/working/` limit. You must implement a `--save-every` parameter combined with a `--keep-last 3` auto-pruning mechanism so only the newest 3 checkpoints are kept.
- **Data & Code Ingestion:** Kaggle's web uploader breaks on raw folders. Assume the human will upload the dataset, the codebase, and the checkpoints as three separate `.zip` files (created with standard forward slashes, not Windows backslashes) mounted as Kaggle Datasets at `/kaggle/input/`.
- **Statelessness:** No persistent local disk between sessions. Resuming runs must dynamically locate the newest `step_*.pth`/`.npz` from a mounted checkpoint dataset.

YOUR TASKS:

1. SETUP & BOILERPLATE CODE (The Kaggle Cell Structure)
   Write the notebook with distinct cells matching standard Kaggle background-run workflows:
   - **Cell 1 (Setup):** Recursively search `/kaggle/input/` for the codebase zip contents, copy it to `/kaggle/working/`, install dependencies (`!pip install -q -e .`), and strictly assert that the GPU is available.
   - **Cell 2 (Data Mount):** Recursively locate the histopathology dataset and previous checkpoints in `/kaggle/input/` and copy them to `/kaggle/working/` so they are accessible and writable if needed.
   - **Cell 3 (Pre-flight Benchmark):** A "smoke test" cell that runs ~10 steps of the pipeline (data loading, augmentations, 1 forward/backward pass) and calculates estimated time-per-epoch to ensure the pipeline isn't bottlenecked by CPU operations before committing to a 10-hour run.

2. IMPLEMENT EXPERIMENT AXES (Modular & Config-Driven)
   - Implement the normalizations, augmentations, and backbones defined in `docs/PLAN.md`.
   - Ensure these are selectable via a clean configuration block or CLI flags callable from the notebook. 
   - Ensure pretrained weights for foundation models/vision backbones either download gracefully (if "Internet on" is assumed) or provide clear instructions on how to mount them as a Kaggle dataset if they require authentication.

3. CHECKPOINTING & TIME-BOXED TRAINING
   - Implement the 10.5-hour time-boxed training loop. The training loop must monitor the wall clock and cleanly break the epoch, save weights, save optimizer state, and exit if 630 minutes is reached.
   - Implement checkpoint pruning (keeping only the top 3 best or newest checkpoints) to strictly respect the 20GB limit.
   - Ensure the training script auto-resumes seamlessly if it detects existing checkpoints in the designated directory, parsing the epoch/step directly from the checkpoint file.

4. LOGGING & EVALUATION
   - Fix random seeds for reproducibility.
   - Log metrics (accuracy, Macro-F1, AUROC, expected calibration error) and domain-gap/robustness drops to a structured `evaluation_result.txt` or CSV file saved to `/kaggle/working/`.
   - Ensure no data leakage occurs across the domain boundary specified in `docs/PLAN.md`.

OUTPUT — write these 4 files to disk:

FILE 1: `docs/kaggle_guide.md` — The exact setup guide the human will follow. Must include:
   - **The Zipping Rule:** Instructions to use a python script (or standard Linux `zip`) with forward slashes for the codebase and dataset, explicitly warning against Windows built-in zip tools.
   - **Dataset Setup:** How to upload the codebase, dataset, and checkpoints as 3 separate Kaggle datasets.
   - **Updating Checkpoints:** Crucial instructions on how to use Kaggle's "New Version" button on the existing checkpoint dataset between sessions, rather than creating a new dataset from scratch every time a 10.5-hour session ends.
   - **Execution Instructions:** How to click "Save Version -> Save & Run All (Commit)" to trigger the 12-hour background run, and the importance of downloading outputs immediately after completion.

FILE 2: `docs/RUN_LOG.md` — a template with one section per planned ablation cell, containing placeholders for: Kaggle notebook link/version, date run, exact config used, status, and notes.

FILE 3: `docs/RESULTS.md` — a results table matching `docs/PLAN.md`'s ablation matrix structure, with metric columns defined but left blank/marked "pending".

FILE 4: All Python source code (`.py`) and Kaggle notebook files (`.ipynb`) placed in a `kaggle/` folder. The code must implement the time-boxed loop, the 20GB-safe checkpoint pruning, and the domain-robustness metrics.

Do not write the final research narrative, discussion, or conclusions. Do not invent evaluation metrics for runs that haven't happened yet.

When done, summarize in chat: how the short-session workflow prevents the 12-hour Kaggle hard kill, how you handled checkpoint pruning for the 20GB limit, and what 3 zip datasets the human needs to upload.
```

---

## AGENT 3 — VERIFICATION (Jules)

```
You are the verification agent, working independently from a clean checkout. Your job is to audit the implementation and results against the original plan — you are the check on the implementation agent's work, not a collaborator with it. Be skeptical by default.

CONTEXT: Read, in this order: `docs/PLAN.md` (what was supposed to happen), `docs/dataset_card.md` (what dataset/split was supposed to be used), `docs/RUN_LOG.md` and `docs/RESULTS.md` (what the implementation agent claims happened), and the actual code/configs/results files in the repository.

YOUR TASKS — verify each of the following independently, don't just trust docs/RUN_LOG.md's claims:

1. PLAN FIDELITY
   - Does the implemented code actually match docs/PLAN.md's experiment design? Check each ablation axis (normalization methods, augmentation policies, backbones) is implemented as specified, not simplified or substituted.
   - Does the dataset/split implementation match docs/dataset_card.md exactly (correct dataset, correct class definitions, correct split logic — especially domain/patient-level split, not accidentally random)?

2. CORRECTNESS AUDIT
   - Check for data leakage: confirm no overlap between train/val/test sets, and specifically no leakage across the domain boundary the robustness claim depends on.
   - Check the evaluation metrics are computed correctly against the stated protocol (right averaging for multi-class, right handling of class imbalance, AUROC computed correctly for the given class setup).
   - Re-derive at least one reported number if feasible (e.g. recompute accuracy from a saved confusion matrix or predictions file) to confirm docs/RESULTS.md isn't misreporting.
   - Check that "robustness" is actually being measured as an in-domain vs out-of-domain comparison, not just aggregate accuracy.

3. RESULTS SANITY
   - Are the results directionally plausible given ML literature on stain normalization (e.g. does normalization help more when domain gap is larger)? Flag any result that looks suspicious (suspiciously perfect scores, suspiciously flat ablation with no differences, metrics that don't move despite a meaningful methodological change).
   - Inspect the saved sample images (results/samples/) if present — do augmented/normalized images look correct, or is there an implementation bug visible by eye (e.g. broken color channels, no visible augmentation effect)?
   - Check whether all ablation matrix cells specified in docs/PLAN.md were actually run; list any gaps.

4. REPRODUCIBILITY CHECK
   - Attempt to reproduce at least one experiment cell from the exact command/config in docs/RUN_LOG.md. Report whether it reproduces the claimed metric within reasonable tolerance.

5. GAP ANALYSIS AGAINST ORIGINAL GOALS
   - Re-read the two original project goals (classify robustly under staining variation; investigate effect of normalization/augmentation/pretrained models). For each, state explicitly whether the completed work actually answers it, partially answers it, or fails to answer it, and why.

OUTPUT — write `VERIFICATION_REPORT.md` with sections:
- Plan Fidelity (pass/fail per axis, with evidence)
- Correctness Issues Found (list, severity-ranked: blocking / significant / minor)
- Results Sanity (any flagged anomalies)
- Reproducibility (what you reproduced and whether it matched)
- Goal Coverage (does the work answer the original two goals — yes/partial/no, with justification)
- Recommended Next Actions (concrete, e.g. "re-run cell X with corrected split", "add missing backbone Y", "no action needed")

Do not fix issues yourself — report them precisely enough that the implementation agent or the human can act on them without re-investigating. If everything checks out, say so explicitly rather than inventing issues to seem thorough.

When done, summarize in chat: overall verdict (plan followed correctly / issues found), the single most important issue if any, and goal coverage status, in under 100 words.
```

---

## How to use this loop

1. Paste Agent 1's prompt into Antigravity with Gemini 3 Pro. Review `docs/PLAN.md` and `docs/dataset_card.md` before proceeding — this is the cheapest point to catch a bad dataset/experiment choice.
2. Paste Agent 2's prompt into your DeepSeek-backed agent, pointing it at the same repo. Let it run the matrix.
3. Paste Agent 3's prompt into Jules against the resulting GitHub repo/PR. Treat `VERIFICATION_REPORT.md` as a gate — if it flags blocking issues, send those back to Agent 2 rather than to Agent 1.
4. Only loop back to Agent 1 if verification reveals the *plan* itself was flawed (e.g. domain gap too small to show any effect), not if it's just an implementation bug.