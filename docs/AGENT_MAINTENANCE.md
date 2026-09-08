# AGENTS.md Maintenance Guide

This file defines how future agents should maintain the repository's root `AGENTS.md`.

The goal is to keep persistent agent context small, stable, accurate, and useful across many tasks.

## Purpose of AGENTS.md

`AGENTS.md` is a compact set of persistent instructions for agents working in this repository.

It is not:

- a project log;
- an experiment journal;
- a changelog;
- a scratchpad;
- a full architecture document;
- a metric report;
- a dataset inventory;
- a record of every completed task.

Treat every line added to `AGENTS.md` as context that future agents may repeatedly receive.

## What belongs in AGENTS.md

Add or keep information when it is durable and affects future agent behavior.

Good candidates include:

- stable project invariants;
- repository-wide conventions;
- safety constraints;
- rules that protect data, experiments, Git history, or API compatibility;
- the current high-level project priority;
- the current default artifact or checkpoint when that choice materially affects normal work;
- authoritative-source routing;
- important repository boundaries;
- rules for expensive or destructive operations;
- validation conventions that apply broadly.

Examples:

- split datasets only by source video/study;
- never overwrite completed experiment runs;
- do not start training without explicit user instruction;
- `train.py` is the source of truth for training behavior;
- web requirements live in `agent_web.md`;
- local datasets must remain outside Git.

## What does not belong in AGENTS.md

Do not add information merely because it was useful in the current task.

Normally keep these elsewhere:

- epoch-by-epoch results;
- complete run histories;
- detailed per-class metrics;
- exact dataset row/frame counts;
- temporary debugging findings;
- one-off shell commands;
- transient implementation details;
- copied documentation;
- long lists of paths that matter to only one workflow;
- speculative future plans;
- completed-task summaries;
- dependency versions already defined authoritatively elsewhere;
- hyperparameter values already defined in code/config;
- facts that are likely to change after the next experiment or import.

If such information matters, put it in a dedicated report or documentation file and route agents to it from `AGENTS.md` only when necessary.

## Prefer authoritative references over duplication

Do not maintain the same fact in multiple places.

Prefer:

> `BEST_MODEL_DIAGNOSTICS.md` is the source of truth for current model metrics and limitations.

over copying its metric tables into `AGENTS.md`.

Prefer:

> `train.py` is the source of truth for training hyperparameters.

over duplicating batch size, dropout, learning rate, weight decay, or augmentation values in persistent instructions.

Prefer:

> Read `EXTERNAL_DATASETS.md` for current dataset import rules and inventory.

over reproducing the complete Galar state in the root file.

When a value can be derived reliably from code, config, tests, manifests, or generated reports, point to that source instead of copying the value.

## Keep documentation lazy-loaded

Agents should not read every project document at session start.

When linking a document from `AGENTS.md`, state when it is relevant.

Good:

- `agent_web.md` — read for web/API/UI tasks.
- `BEST_MODEL_DIAGNOSTICS.md` — read for evaluation or checkpoint decisions.
- `EXTERNAL_DATASETS.md` — read for dataset import or metadata tasks.

Bad:

> Read all files in `docs/` before doing any task.

A document that is unrelated to the current task should remain outside the active context.

## Decision test before adding a line

Before adding persistent information to `AGENTS.md`, ask:

1. Will this still matter several tasks from now?
2. Can violating it cause incorrect work or damage?
3. Does it affect many tasks rather than only the current one?
4. Is there already an authoritative source for this information?
5. Could a short reference replace the detailed content?
6. Is this likely to become stale soon?

If the information is temporary, duplicated, task-specific, or volatile, do not add it.

## Updating current project state

Keep only high-level state that changes how agents should act.

Good:

> Training is paused. Do not start new training runs unless explicitly requested.

Good:

> The active deliverable is the local web MVP.

Bad:

> Run A stopped at epoch 7, run B stopped at epoch 10, run C scored 0.294...

Detailed historical state belongs in reports.

When the project priority changes, replace the old priority. Do not append a chronological history to `AGENTS.md`.

## Handling experiment results

Do not add routine experiment results to `AGENTS.md`.

Experiment results should live in dedicated artifacts such as:

- model diagnostics;
- experiment reports;
- run summaries;
- generated JSON/CSV reports;
- a dedicated model-state document if needed.

Only promote an experiment result into `AGENTS.md` when it creates a durable behavioral rule.

Example:

A new run becomes the selected default checkpoint.

Then `AGENTS.md` may update the default checkpoint path, while detailed metrics remain in the diagnostic report.

## Handling dataset changes

Do not store changing dataset inventories or counts in `AGENTS.md`.

Keep durable semantic rules only, for example:

- preserve source IDs;
- split by source study/video;
- exclude conflicting pathological targets;
- never map generic bleeding to `blood_hematin`.

Exact counts, missing-study lists, download status, and import summaries belong in dataset documentation, manifests, or generated reports.

## Handling temporary repository state

Persistent warnings about dangerous working-tree state may belong in `AGENTS.md` when future agents could accidentally destroy or misrepresent data.

Examples:

- intentionally dirty tracked artifacts;
- completed runs that must not be overwritten;
- local files that must not be committed.

Remove or update these warnings as soon as the underlying condition changes.

Do not leave resolved incidents in the file as historical notes.

## Commands

Keep only commands or command conventions that are broadly useful and safe.

Do not accumulate:

- one-off investigation commands;
- old experiment launch commands;
- temporary migration commands;
- commands copied from previous tasks.

Complex workflow commands belong next to the workflow documentation.

Commands that trigger training, large downloads, bulk extraction, destructive migrations, or expensive processing should not be presented as routine validation.

## Agent-efficiency requirements

When editing `AGENTS.md`, preserve context efficiency.

Prefer:

- short declarative rules;
- references to authoritative files;
- task-specific lazy loading;
- replacing obsolete lines;
- grouping closely related rules.

Avoid:

- repeated explanations;
- prose history;
- duplicate facts;
- long examples when a short invariant is sufficient;
- exact values that are already defined elsewhere;
- instructions to scan or preload the repository.

The root file should remain useful without becoming a complete project encyclopedia.

## Procedure for modifying AGENTS.md

Before a non-trivial edit:

1. Read the existing relevant section.
2. Identify whether the new information is durable enough for persistent context.
3. Check whether an authoritative source already exists.
4. Prefer updating or replacing an existing line over appending another one.
5. Move detailed supporting information to an appropriate dedicated document.
6. Remove obsolete or contradictory instructions.
7. Check for duplicate facts elsewhere in the file.
8. Keep wording as short and operational as possible.
9. Ensure references state when the linked document should be read.
10. Verify that the resulting file still reflects the current project priority and safety constraints.

## After completing a task

Do not update `AGENTS.md` automatically just because work was completed.

Update it only when the task changed one of these:

- a durable invariant;
- a repository-wide convention;
- a safety rule;
- an authoritative source;
- a long-lived project priority;
- a default artifact/checkpoint used by future work;
- an important workflow boundary.

Otherwise leave `AGENTS.md` unchanged.

## Removing information

Deletion is part of maintenance.

Remove an instruction when:

- it is no longer true;
- the relevant risk no longer exists;
- the project priority changed;
- another source became authoritative;
- the rule was temporary;
- it duplicates a better instruction;
- it no longer affects agent behavior.

Do not preserve obsolete information for historical completeness. History belongs elsewhere.

## Size and quality target

There is no strict line limit, but the root file should remain compact enough to be cheap persistent context.

If `AGENTS.md` begins accumulating detailed metrics, inventories, long commands, experiment histories, or workflow-specific documentation, split those sections into dedicated files.

Optimize for:

1. correctness;
2. safety;
3. stable behavioral guidance;
4. clear routing to sources of truth;
5. minimal persistent context.

Do not optimize `AGENTS.md` for completeness.
