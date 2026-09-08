# Project

The repository contains the established Kvasir-Capsule 14-class frame
classifier and an isolated experimental Galar dual-model pipeline.

`train.py` is the canonical pipeline for the established 14-class task.
`galar_dual_model/` is the source tree for the independent Galar Anatomy and
Pathology tasks; do not merge its behavior into `train.py` unless explicitly
requested.
The verified Windows environment uses `.venv-win`, Python 3.14.2, PyTorch 2.13.0+cu130, CUDA, and an RTX 3060 12 GB.
Use `--workers 0` for Windows training/data-loader workflows unless the relevant code or documentation explicitly says otherwise.

## Sources of truth

Read additional documentation only when it is relevant to the current task.

- `agent_web.md` — web MVP requirements, API contract, UI behavior, tests, and safety wording. Read for web/API/UI/inference-demo tasks.
- `BEST_MODEL_DIAGNOSTICS.md` — selected checkpoint, model comparisons, per-class metrics, characteristic errors, and known limitations. Read for model evaluation, checkpoint decisions, or claims about model quality.
- `EXTERNAL_DATASETS.md` — external dataset sources, mappings, import rules, and provenance requirements. Read for dataset import, metadata, Galar, Capsule Vision, SEE-AI, or KID tasks.
- `train.py` — source of truth for the established 14-class training pipeline and hyperparameters.
- `galar_dual_model/README.md` — Galar dual-model task definitions, metadata/cache workflows, training commands, outputs, and limitations. Read for Anatomy/Pathology training or Galar dual-model operations.
- `galar_dual_model/labels.py` — source of truth for the Galar Anatomy, Pathology, technical, and zero-support label groups.
- `galar_dual_model/prepare_metadata.py` and `galar_dual_model/train_common.py` — sources of truth for the dual-model split/sampling policy and training behavior respectively.
- `docs/AGENT_MAINTENANCE.md` — rules for maintaining this file. Read before making non-trivial changes to `AGENTS.md`.

Do not preload these documents when they are unrelated to the current task.

## ML and data invariants

- Split strictly by `video_id` or source study/video ID.
- Never randomly split neighboring frames from the same source video between train and validation.
- Model selection must use macro-F1 over classes represented in unseen-video validation.
- Do not use same-video temporal holdout results as evidence of cross-video generalization.
- Exact duplicates and frames with conflicting pathological target labels are excluded from the current single-label softmax task.
- The Galar Anatomy and Pathology tasks are multi-label. Preserve valid label co-occurrence, use independent logits with sigmoid/BCE-style training, and never force these tasks into softmax or discard multi-label frames for convenience.
- Keep Galar anatomical, pathological, and technical labels separate according to `galar_dual_model/labels.py`; do not invent or infer new classes from names alone.
- Preserve source study/video IDs when importing external data.
- Never map generic `bleeding`, `blood`, or `active bleeding` to `blood_hematin`.
- Galar `papilla of Vater` / `ampulla of vater` maps to `ampulla_of_vater`; `hematin` maps to `blood_hematin`; `polyp` maps to `polyp`.
- Training-only temporal caps must not reduce validation video coverage.
- Do not use diagnostic-only results for checkpoint selection.

## Current priority

The active workstream is the isolated Galar dual-model pipeline. The web MVP is
postponed: do not work on `agent_web.md`, frontend code, or web MVP tasks unless
the user explicitly resumes that workstream.

The user-authorized Anatomy and Pathology experiments are complete, and no
dual-model training process is currently active. Anatomy did not satisfy its
continuation gate, and the completed Pathology follow-ups did not replace the
existing fixed-threshold baseline. Do not resume these runs automatically;
preserve their checkpoints, histories, and run configurations. Treat
threshold-tuning output as diagnostic-only, never as independent validation or
checkpoint-selection evidence.

Do not start, stop, restart, or duplicate an expensive training run without
explicit authorization. Before launching a run, check for an existing training
process and a non-empty output directory. Never overwrite an in-progress,
interrupted, or completed Galar run; use a new output directory. Use
`--workers 0` for subsequent Galar training on Windows unless the user
explicitly requests multiprocessing after available commit/pagefile capacity
has been verified.

Keep all new dual-model implementation inside `galar_dual_model/`. Reuse ideas
from the established pipeline where useful, but do not modify old experiment
runs or the established pipeline merely to support the dual-model work.

The current Galar baseline uses EfficientNet-B0 transfer learning. Exact
augmentation, staged freeze/unfreeze, optimizer, loss, early-stopping, and
checkpoint-selection behavior belongs in `galar_dual_model/train_common.py`,
not in this file.

The local SSD image cache, generated metadata, and model outputs are runtime
artifacts. They must remain outside Git, and training from the cache must not
require the original dataset drive once cache coverage has been validated.

The local web MVP remains deferred, but its default inference checkpoint is:

`runs/combined_galar_head_only/best.pt`

This is the formal metric-best checkpoint currently selected for the demo.

The web MVP must not require either local dataset at runtime. Only the checkpoint and required report/config artifacts may be runtime dependencies.

## Web/model safety

For web behavior and wording, follow `agent_web.md` and `BEST_MODEL_DIAGNOSTICS.md`.

At minimum:

- call softmax outputs "model scores", not diagnostic probabilities;
- show the top-3 model outputs;
- clearly state that the frame classifier is a research prototype without temporal context;
- do not hide, soften, or overstate known model limitations;
- do not present model output as a medical diagnosis.

## Working-tree safety

Tracked files under `runs/regularized_v2` may contain intentionally uncommitted artifacts from an accidental one-epoch overwrite.

Do not stage, commit, evaluate, restore, discard, or present those dirty local artifacts as valid historical results unless the user explicitly requests it.

The committed `HEAD` versions are the authoritative historical artifacts for that run.

Completed experiment directories must not be overwritten:

- `runs/combined_galar_v1`
- `runs/combined_galar_head_only`
- `runs/combined_galar_last_block`

Treat every non-empty directory under `galar_dual_model/runs/` as protected:
inspect it before choosing an output path, and never overwrite or delete it
without an explicit user request.

Downloaded videos, extracted frames, local datasets, raw annotation trees, `.venv-win`, and other large/generated data artifacts must remain outside Git unless explicitly documented otherwise.

## Repository exploration

Start with the smallest relevant scope.

- Prefer targeted search and targeted file reads.
- Inspect the minimum number of files needed to establish the relevant code path.
- Expand scope only when the current evidence is insufficient.
- Do not scan the entire repository unless the task genuinely requires it.
- Do not reread files that were already inspected unless they may have changed or a specific unresolved question requires it.
- Do not inspect datasets, model weights, `.venv-win`, raw frames, generated artifacts, or completed run contents unless they are directly relevant.
- Do not dump large directories, lockfiles, generated files, or binary metadata into context without a task-specific reason.
- Read optional documentation only when relevant to the task.

## Implementation rules

- Prefer minimal, task-focused diffs.
- Do not refactor unrelated code.
- Reuse existing project utilities and conventions before introducing new abstractions.
- Do not add dependencies unless they are necessary.
- Preserve existing API contracts unless the task explicitly changes them.
- Treat code, configuration, tests, and generated reports as sources of truth instead of duplicating their values in `AGENTS.md`.
- Do not launch expensive training, bulk extraction, large downloads, destructive migrations, or other high-cost operations unless explicitly requested.
- When exact dataset counts, metrics, dependency versions, or hyperparameters are needed, read them from the current authoritative artifact rather than assuming values from memory.

## Validation

Run the narrowest relevant validation first and expand only if needed.

For Python commands on Windows, prefer the project environment:

`.\\.venv-win\\Scripts\\python.exe`

For web-specific validation, follow `agent_web.md`.

For Galar dual-model validation, start with import/compile checks and the
documented `--dry-run` workflow. A dry run must use a separate or non-writing
path and must not disturb an active training process.

A dry run is not permission to start training. Do not convert validation into a real training run unless explicitly requested.

## Maintaining this file

Keep `AGENTS.md` limited to persistent invariants, safety rules, current high-level priorities, authoritative-source routing, and repository-wide working conventions.

Do not use it as an experiment log, changelog, task journal, metric report, or dataset inventory.

Before making substantial changes to this file, follow `docs/AGENT_MAINTENANCE.md`.

Prefer replacing obsolete information over appending history.

## Final response

Keep final responses concise.

Report only what is useful:

- what changed;
- which files changed;
- validation performed and its result;
- unresolved issues or important limitations, if any.

Do not include a long walkthrough unless the user asks for one.
