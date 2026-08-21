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
- The previous run completed with early stopping at epoch 9. Its best checkpoint was epoch 5,
  supported validation macro-F1 `0.2428336`; same-video diagnostic macro-recall `1.0` (not
  cross-video evidence). It predates the anti-overfitting changes and remains in `runs/baseline`.
  The new dry-run is in `runs/anti_overfit_audit`; the regularized model has not been trained yet.

## Commands

```powershell
.\.venv-win\Scripts\python.exe train.py --dry-run --output runs\audit
.\.venv-win\Scripts\python.exe train.py --output runs\regularized_v2 --epochs 15 --workers 0
```

Preserve video-level isolation when adding external data. Do not use diagnostic results for model selection.
