# Project context

Kvasir-Capsule 14-class frame classifier using EfficientNet-B0. `train.py` is the canonical pipeline; `README.md` has setup details. Windows/CUDA environment: `.venv-win`, `--workers 0`. Default data paths are `D:\dataset_quazir\labelled_images` and `D:\dataset_quazir\metadata.csv`.

## Current state

- Default batch size is 16.
- Split strictly by `video_id`; never randomly split neighboring frames.
- Select checkpoints by macro-F1 over classes present in video-level validation.
- `ampulla_of_vater`, `blood_hematin`, and `polyp` each occur in one source video. Their temporal holdout is `diagnostic_only`, never evidence of cross-video generalization.
- Anti-overfitting defaults: cap each `(video_id, label)` sequence at 500 evenly spaced
  training frames, stronger orientation-safe augmentation, dropout `0.4`, label smoothing
  `0.1`, weight decay `0.01`, and sampler multiplier cap `10`.
- Freeze the backbone for two epochs, then train only its final 3/9 blocks at `0.1x` the
  classifier learning rate. Frozen BatchNorm statistics stay frozen.
- Exact duplicates and conflicting multi-label frames are audited and excluded from this single-label softmax task.
- The original run stopped at epoch 9. Its best checkpoint is still the project best: epoch 5,
  supported validation macro-F1 `0.2428336`; artifacts are in `runs/baseline`.
- The regularized run ended unexpectedly after epoch 7. Its best checkpoint is epoch 4,
  supported validation macro-F1 `0.2219359`; artifacts are in `runs/regularized_v2`. It did not
  reach final diagnostic evaluation, so no `diagnostic_report.json` is expected for that run.
- `runs/anti_overfit_audit` is the completed dry-run for the regularized pipeline.

## External data

- Primary choice: Galar, mapping `papilla of Vater -> ampulla_of_vater`,
  `hematin -> blood_hematin`, and `polyp -> polyp`.
- Capsule Vision 2024 / SEE-AI / KID are secondary sources for `polyp`; exclude every row whose
  source is Kvasir to avoid duplicates. Never map generic `bleeding` to `blood_hematin`.
- Preserve source study/video IDs and split only by them. Galar is multi-label; ignore technical
  and GI-section labels, but exclude frames with conflicting pathological targets for the current
  single-label model. Full links and import rules are in `EXTERNAL_DATASETS.md`.

## Commands

```powershell
.\.venv-win\Scripts\python.exe train.py --dry-run --output runs\audit
.\.venv-win\Scripts\python.exe train.py --output runs\regularized_v2 --epochs 15 --workers 0
```

Preserve video-level isolation when adding external data. Do not use diagnostic results for model selection.
