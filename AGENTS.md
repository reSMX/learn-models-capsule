# Project context

Kvasir-Capsule 14-class frame classifier using EfficientNet-B0. `train.py` is the canonical pipeline; `README.md` has setup details. Windows/CUDA environment: `.venv-win`, `--workers 0`. Default data paths are `D:\dataset_quazir\labelled_images` and `D:\dataset_quazir\metadata.csv`.

## Current state

- Default batch size is 16.
- Split strictly by `video_id`; never randomly split neighboring frames.
- Select checkpoints by macro-F1 over classes present in video-level validation.
- `ampulla_of_vater`, `blood_hematin`, and `polyp` each occur in one source video. Their temporal holdout is `diagnostic_only`, never evidence of cross-video generalization.
- Training uses capped square-root sampling, two frozen-backbone epochs, weight decay, and early stopping.
- Exact duplicates and conflicting multi-label frames are audited and excluded from this single-label softmax task.
- Retraining completed with early stopping at epoch 9. Best checkpoint: epoch 5, supported validation macro-F1 `0.2428336`; same-video diagnostic macro-recall `1.0` (not cross-video evidence). Current artifacts are in `runs/baseline`; dry-run results are in `runs/audit_check`.

## Commands

```powershell
.\.venv-win\Scripts\python.exe train.py --dry-run --output runs\audit
.\.venv-win\Scripts\python.exe train.py --output runs\baseline_v2 --epochs 15 --workers 0
```

Preserve video-level isolation when adding external data. Do not use diagnostic results for model selection.
