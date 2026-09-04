# Project context

Kvasir-Capsule 14-class frame classifier using EfficientNet-B0. `train.py` is the canonical pipeline; `README.md` has setup details. Windows/CUDA environment: `.venv-win`, `--workers 0`. The verified environment is Python 3.14.2 with PyTorch 2.13.0+cu130 on an RTX 3060 12 GB (CUDA compute capability 8.6, 28 SMs / 3,584 CUDA cores); CUDA, cuDNN, AMP, and a real FP32 tensor operation all pass. Default data paths are `D:\dataset_quazir\labelled_images` and `D:\dataset_quazir\metadata.csv`.

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
- The local Kvasir metadata has 47,248 raw rows. The completed audit removes 10 exact duplicate
  rows and 9 conflicting multi-label frames (18 rows), leaving 47,220 usable single-label rows.
  After cleanup, `ampulla_of_vater`, `blood_hematin`, and `polyp` contain 9, 12, and 55 frames,
  respectively, and still come from only one Kvasir video each.
- The original Kvasir-only run stopped at epoch 9. Its best checkpoint is epoch 5,
  supported validation macro-F1 `0.2428336`; artifacts are in `runs/baseline`.
- The regularized run ended unexpectedly after epoch 7. Its best checkpoint is epoch 4,
  supported validation macro-F1 `0.2219359`; artifacts are in `runs/regularized_v2`. It did not
  reach final diagnostic evaluation, so no `diagnostic_report.json` is expected for that run.
- `runs/anti_overfit_audit` is the completed dry-run for the regularized pipeline.
- `runs/combined_galar_audit` is the completed combined-data dry-run: 71,609 raw rows,
  71,581 usable single-label rows, train/validation sizes 28,117/18,037 after the train-only
  temporal cap, no missing images, no diagnostic-only classes, and all 14 classes represented in
  unseen-video validation.
- `runs/combined_galar_v1` completed by early stopping after epoch 7. Its best checkpoint is epoch
  3 with supported validation macro-F1 `0.2961812`; all 14 classes have validation support, so no
  diagnostic-only report is needed. This is the current project-best checkpoint.

## External data

- Primary choice: Galar, mapping `papilla of Vater -> ampulla_of_vater`,
  `hematin -> blood_hematin`, and `polyp -> polyp`.
- Capsule Vision 2024 / SEE-AI / KID are secondary sources for `polyp`; exclude every row whose
  source is Kvasir to avoid duplicates. Never map generic `bleeding` to `blood_hematin`.
- Preserve source study/video IDs and split only by them. Galar is multi-label; ignore technical
  and GI-section labels, but exclude frames with conflicting pathological targets for the current
  single-label model. Full links and import rules are in `EXTERNAL_DATASETS.md`.
- The local Galar release is in `D:\Dataset_galar`: `metadata.csv`, 80 annotation CSVs under
  `Labels/`, and official pre-extracted PNG frame folders. The download contains studies 1-40 and
  61-80; study folders 41-60 are unavailable because two upstream archives could not be downloaded.
  Do not treat an entire study as the target class; use only rows whose target column is `1`, using
  the exact `frame_XXXXXX.PNG` source frame from the `frame` column.
- In the local Galar files, the ampulla column is named `ampulla of vater`. The importer maps it to
  `ampulla_of_vater`; it maps `hematin` to `blood_hematin` and `polyp` to `polyp`. GI-section and
  anatomical landmark labels may coexist with a target. Other pathological labels make that frame
  ambiguous and exclude it from the current single-label task. Generic `blood` or `active bleeding`
  must not be converted to `blood_hematin`.
- `select_galar_target_videos.py` scans the annotations and writes
  `another db/galar_target_videos.csv`. The current manifest contains 49 unique Galar videos:
  8 with usable ampulla frames, 22 with usable hematin frames, and 32 with usable polyp frames
  (some videos contain more than one target class). Strictly usable frame totals are 1,252,
  28,571, and 14,567, respectively.
- Of those 49 selected studies, 40 are locally available. The intentionally omitted selected study
  IDs are 43, 44, 47, 53, 54, 56, 57, 59, and 60. The available clean totals are 37 ampulla frames,
  18,917 hematin frames, and 5,407 polyp frames across 40 source studies.
- `extract_galar_target_frames.py` supports both source videos and official pre-extracted PNG
  folders. The completed PNG import copied 24,361 clean target frames to
  `D:\dataset_quazir\galar_target_images`, preserved IDs such as `galar_5`, wrote separate Galar
  metadata and `D:\dataset_quazir\metadata_with_galar.csv`, and recorded 3,148 ambiguous exclusions.
  The original Kvasir metadata and original Galar PNGs were not overwritten.
- The extractor keeps all clean frames by default. Let `train.py` apply its 500-frame cap only to
  training groups so validation video coverage is preserved. If storage forces an extraction cap,
  use `--max-frames-per-video-class 500` explicitly.
- Do not commit downloaded videos, extracted frames, or the 246 MB raw `another db/Labels` tree.
  The small filtered manifest is safe to version.

## Commands

```powershell
.\.venv-win\Scripts\python.exe train.py --dry-run --output runs\audit
.\.venv-win\Scripts\python.exe train.py --output runs\regularized_v2 --epochs 15 --workers 0
.\.venv-win\Scripts\python.exe select_galar_target_videos.py
# Import the available official Galar PNG folders; absent studies are recorded in the summary:
.\.venv-win\Scripts\python.exe extract_galar_target_frames.py --frames-root "D:\Dataset_galar" --galar-root "D:\Dataset_galar" --skip-missing-studies
.\.venv-win\Scripts\python.exe train.py --images "D:\dataset_quazir" --metadata "D:\dataset_quazir\metadata_with_galar.csv" --output runs\combined_galar_audit --workers 0 --dry-run
.\.venv-win\Scripts\python.exe train.py --images "D:\dataset_quazir" --metadata "D:\dataset_quazir\metadata_with_galar.csv" --output runs\combined_galar_v2 --epochs 15 --workers 0
```

When training the combined dataset, point `--images` at `D:\dataset_quazir` so both the original
and external class folders are indexed, and point `--metadata` at
`D:\dataset_quazir\metadata_with_galar.csv`. Preserve video-level isolation when adding external
data. Do not use diagnostic results for model selection.
