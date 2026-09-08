# Galar dual-model baseline

This directory is an isolated experimental pipeline for two independent
EfficientNet-B0 models. It does not modify `train.py` or any existing run.

## Label audit and task definitions

The source is the 80 official `Labels/*.csv` files (3,513,539 annotated frame
rows). Labels are kept with their original Galar names.

**Anatomy** uses nine labels:

- landmarks: `z-line`, `pylorus`, `ampulla of vater`, `ileocecal valve`;
- GI sections: `mouth`, `esophagus`, `stomach`, `small intestine`, `colon`.

Anatomy is **multi-label**. The source annotations contain 8,737 frames on
which a landmark and its GI section are both valid (for example, `ampulla of
vater` + `small intestine`). Treating all nine outputs as a softmax would erase
that hierarchy. The model therefore emits nine independent logits and uses
sigmoid at inference.

**Pathology** uses every pathological column with positive support:

`ulcer`, `polyp`, `active bleeding`, `blood`, `erythema`, `erosion`,
`angiectasia`, `IBD`, `foreign body`, `hematin`, `cancer`,
`lymphangioectasis`.

Pathology is **multi-label**. There are 24,481 multi-pathology frames in the
complete annotation set. Frames with no positive pathology label are retained
as all-zero negative examples; generic `blood` and `active bleeding` remain
separate from `hematin`.

Excluded labels/columns:

- `bubbles`, `dirt`, `no view`, `reduced view`, `good view`: technical image
  quality, not anatomy or pathology;
- `index`, `frame`: row/frame identifiers;
- `section`: a text duplicate of the five binary GI-section columns (apart
  from one source row containing `0`);
- `esophagitis`, `varices`, `celiac`: pathological schema columns with zero
  positive frames in all 80 annotation CSVs, so they cannot be learned or
  validated.

## Metadata and split

`prepare_metadata.py` first scans annotations by source study, then chooses one
shared deterministic split for both tasks. With the defaults, 20% of complete
studies are validation studies and the seed is 42. Candidate splits are scored
for per-class frame/study balance and are rejected if a supported class would
be absent from train or, where at least two source studies exist, validation.
No frame-level random split is performed.

Training metadata is temporally thinned after the study split; validation is
never thinned:

- Anatomy retains every landmark frame and up to 1,000 evenly spaced frames
  per `(study, exact section-only labelset)`;
- Pathology retains every frame containing a rare class, up to 1,000 evenly
  spaced frames per `(study, common positive labelset)`, and up to 1,000
  evenly spaced all-zero negative frames per study;
- a pathology class is considered rare when the raw training split contains
  at most 5,000 positive frames or at most three positive studies.

This removes long near-duplicate sequences without removing a study, class, or
validation frame. The exact policy, raw counts, retained counts, and preserved
rare classes are stored in each task config and `summary.json`. The caps can be
changed with `prepare_metadata.py` arguments; setting a cap to `0` disables it.

For the currently available 60-study local subset, the defaults retain 121,556
of 881,713 Anatomy training frames (1,900 batches at the default batch size 64)
and 106,398 of 881,714 Pathology training frames (1,663 batches). All 143,291
validation frames remain unchanged for both tasks.

Generated files under `galar_dual_model/metadata/` are:

- `anatomy_train.csv`, `anatomy_validation.csv`;
- `pathology_train.csv`, `pathology_validation.csv`;
- `anatomy_config.json`, `pathology_config.json`;
- `split.json`, `summary.json`.

Each task CSV has `image_path`, `study_id`, `video_id`, `frame_number`, and a
pipe-separated `labels` field. Image paths are relative to `--galar-root`.
The JSON summary records class frame/study counts, split sizes, multi-label and
negative-frame counts, missing validation support, excluded technical labels,
and unavailable local PNG studies. The generated metadata and run directories
are ignored by Git.

The local copy currently has annotation CSVs for all 80 studies but PNG folders
for studies 1-40 and 61-80 only. `--missing-study-policy skip` makes this
limitation explicit and prepares the 60 locally trainable studies. Once the
other PNG directories are downloaded, omit that argument (the default is to
fail rather than silently omit studies). `--verify-images` optionally checks
every referenced PNG and is intentionally not the default because it performs
millions of filesystem calls.

## SSD image cache

`prepare_image_cache.py` reads the union of all four generated task CSVs, so a
frame needed by both models is stored only once. With the current metadata this
is 348,657 unique frames. Each source PNG is resized to 256x256 and encoded as
JPEG quality 95 with 4:4:4 chroma (no colour subsampling). The measured payload
estimate is about 7.8 GiB.

The cache does not create hundreds of thousands of individual image files.
JPEG byte streams are concatenated into one seekable shard per source study,
with a small fixed-record frame index. This avoids repeated filesystem-open
latency while retaining source `study_id` and `frame_number`. A manifest stores
the exact resize/encoding settings and SHA-256 fingerprints of the generated
metadata. Training refuses a stale cache if metadata is regenerated.

The conversion copies and transcodes required frames; it never deletes or
changes the original Galar PNGs. Free space is checked throughout the build and
the default `--minimum-free-gb 15` reserves at least 15 GiB on the destination
volume. An interrupted build can be continued with the same command plus
`--resume`; only a partially written study is rebuilt.

## Preparation and smoke checks (PowerShell)

From the repository root:

```powershell
.\.venv-win\Scripts\python.exe -B .\galar_dual_model\prepare_metadata.py --galar-root 'D:\Dataset_galar' --missing-study-policy skip --overwrite

.\.venv-win\Scripts\python.exe -B .\galar_dual_model\prepare_image_cache.py --galar-root 'D:\Dataset_galar' --output-root 'C:\Users\Maxx\Galar_256_cache' --workers 4 --minimum-free-gb 15 --dry-run --dry-run-samples 32

.\.venv-win\Scripts\python.exe -B .\galar_dual_model\prepare_image_cache.py --galar-root 'D:\Dataset_galar' --output-root 'C:\Users\Maxx\Galar_256_cache' --image-size 256 --jpeg-quality 95 --jpeg-subsampling '4:4:4' --workers 4 --conversion-batch-size 64 --minimum-free-gb 15

.\.venv-win\Scripts\python.exe -B .\galar_dual_model\train_anatomy.py --image-cache-root 'C:\Users\Maxx\Galar_256_cache' --workers 2 --device cpu --batch-size 2 --ram-buffer-gb 2 --dry-run --dry-run-samples 4

.\.venv-win\Scripts\python.exe -B .\galar_dual_model\train_pathology.py --image-cache-root 'C:\Users\Maxx\Galar_256_cache' --workers 2 --device cpu --batch-size 2 --ram-buffer-gb 2 --dry-run --dry-run-samples 4
```

The cache dry run converts a small real sample in memory, reports the union and
estimated cache size, and writes nothing. Training dry-run mode builds the
model without downloading pretrained weights, loads real cached train and
validation images, and performs a small forward/backward pass. It does not
create checkpoints or other run artifacts.

## Full training (PowerShell)

Do not start these two commands simultaneously on one GPU:

```powershell
.\.venv-win\Scripts\python.exe -B .\galar_dual_model\train_anatomy.py --image-cache-root 'C:\Users\Maxx\Galar_256_cache' --workers 2 --prefetch-factor 2 --batch-size 64 --ram-buffer-gb 2 --read-block-size 512 --device cuda

.\.venv-win\Scripts\python.exe -B .\galar_dual_model\train_pathology.py --image-cache-root 'C:\Users\Maxx\Galar_256_cache' --workers 2 --prefetch-factor 2 --batch-size 64 --ram-buffer-gb 2 --read-block-size 512 --device cuda
```

When `--image-cache-root` is supplied, training does not read the original PNG
tree. `--galar-root 'D:\Dataset_galar'` remains supported as a slower fallback
when no cache is supplied.

Defaults are 20 epochs, a batch size of 64, two spawned DataLoader workers with
two prefetched batches each, two frozen-backbone epochs, AMP on CUDA, and early
stopping after four epochs without improvement. After the frozen warm-up,
EfficientNet is unfrozen gradually from its output backwards: two of its nine
feature stages per epoch (`--unfreeze-stages-per-epoch 2`). The backbone uses a
lower `3e-5` learning rate while the classifier uses `3e-4`; setting
`--unfreeze-stages-per-epoch 0` restores one-step full unfreezing. This worker
setting has been smoke-tested with this dataset class on the verified Windows
environment; use `--workers 0` as the debugging fallback if worker startup
fails.

Training augmentation uses random resized crops (75-100% scale), horizontal and
vertical flips, rotations up to 25 degrees, moderate colour jitter, occasional
light Gaussian blur, and random erasing. Validation uses only deterministic
resize, centre crop, and normalization.

Metric counters remain on the GPU throughout each phase and transfer to CPU
once at the end. The model and input batches use channels-last CUDA layout.
The progress bar refreshes current batch loss every 50 batches rather than
synchronising CPU and GPU on every iteration; change this with
`--loss-display-interval`. The persistent epoch line and history files still
use the exact mean loss over the complete phase. If 12 GB VRAM is insufficient,
lower `--batch-size`.

The default `--ram-buffer-gb 2` is a bounded double buffer in ordinary RAM.
Metadata is read in shuffled blocks of 512 neighboring rows, while file reads
inside a block stay sequential. One half-buffer of decoded/augmented tensors is
consumed by the GPU while a background thread fills the other half through the
DataLoader workers; then the slots are swapped. Samples are shuffled inside
each train buffer, while validation remains ordered. Only the outgoing batch is
page-locked, so the whole 2 GiB cache does not consume pinned memory. The first
phase progress bar remains at zero while its first half-buffer is warmed. Use
`--ram-buffer-gb 0` to disable this layer and return to the regular DataLoader.

Both tasks use `BCEWithLogitsLoss`. Training-split `negatives / positives`
weights compensate for class imbalance and are capped at 100 to avoid extreme
weights for very rare findings. Checkpoint selection uses the mean of binary
per-class F1 scores over classes supported in unseen-study validation at the
fixed threshold `0.5`; this is not multiclass softmax macro-F1. The checkpoint
stores the threshold and class order needed for sigmoid inference.

### Pathology imbalance experiment

The weighted-BCE run remains the baseline. A follow-up Pathology experiment can
select `--loss asymmetric`, which implements the asymmetric focal objective from
the [ICCV 2021 paper and official implementation](https://github.com/Alibaba-MIIL/ASL).
Its separate positive and negative focusing terms are a better match for sparse
multi-label findings than applying large inverse-prevalence weights to every
positive frame. The defaults (`gamma_neg=4`, `gamma_pos=1`, `clip=0.05`) follow
the published configuration. Loss reduction is a mean so the existing learning
rates remain on the same scale as the BCE baseline.

`--initial-backbone-checkpoint` transfers only EfficientNet feature weights from
another compatible Galar checkpoint. In particular, an Anatomy checkpoint can
provide capsule-domain initialization while Pathology keeps a new independent
12-label sigmoid classifier. `--initial-checkpoint` instead warm-starts the
complete model and requires an identical task and class order. Both modes record
their source checkpoint and epoch in `run_config.json`.

The proposed first comparison is:

```powershell
.\.venv-win\Scripts\python.exe -B .\galar_dual_model\train_pathology.py --image-cache-root 'C:\Users\Maxx\Galar_256_cache' --workers 0 --batch-size 64 --ram-buffer-gb 2 --read-block-size 512 --device cuda --loss asymmetric --asymmetric-gamma-neg 4 --asymmetric-gamma-pos 1 --asymmetric-clip 0.05 --classifier-dropout 0.4 --initial-backbone-checkpoint '<anatomy-best.pt>' --output 'galar_dual_model\runs\pathology_asl_anatomy_init'
```

If that run does not beat the fixed-threshold BCE baseline, the next controlled
comparison retains BCE but smooths the inverse-prevalence weights. Setting
`--pos-weight-power 0.5` uses the square root of each negatives/positives ratio;
`0` disables positive reweighting and the baseline value is `1`.

```powershell
.\.venv-win\Scripts\python.exe -B .\galar_dual_model\train_pathology.py --image-cache-root 'C:\Users\Maxx\Galar_256_cache' --workers 0 --batch-size 64 --ram-buffer-gb 2 --read-block-size 512 --device cuda --loss bce --pos-weight-power 0.5 --max-pos-weight 20 --classifier-dropout 0.4 --initial-backbone-checkpoint '<anatomy-best.pt>' --output 'galar_dual_model\runs\pathology_sqrt_bce_anatomy_init'
```

This choice is also consistent with a recent
[Galar-specific imbalanced multi-label pipeline](https://arxiv.org/abs/2603.17879),
which combines asymmetric focal loss with class-aware sampling, mixup, and
per-class threshold calibration. The latter two remain possible follow-ups, not
part of this first controlled comparison. The official
[Galar training repository](https://github.com/EKFZ-AI-Endoscopy/GalarCapsuleML)
also treats weighted loss/sampling, dropout, larger backbones, and study-level
cross-validation as explicit experiment dimensions.

After a run finishes, `evaluate_thresholds.py` can sweep global and per-class
sigmoid thresholds in one validation pass. Its output is explicitly
`diagnostic_only`: tuning and measuring on the same validation studies is
optimistic, so the result must never select a checkpoint or be reported as an
unbiased test score.

```powershell
.\.venv-win\Scripts\python.exe -B .\galar_dual_model\evaluate_thresholds.py --checkpoint '<run>\best.pt' --image-cache-root 'C:\Users\Maxx\Galar_256_cache' --workers 0 --device cuda
```

Loss changes cannot compensate for missing patient diversity. Classes supported
by only one or two train/validation studies should be treated as data-limited;
the highest-leverage follow-up is adding the currently unavailable Galar studies
and rebuilding metadata/cache without changing the unseen-study split invariant.

Training refuses to write into a non-empty output directory. Always choose a new
run name for a follow-up or warm-start experiment.

Outputs are separate:

- Anatomy: `galar_dual_model/runs/anatomy/best.pt`;
- Pathology: `galar_dual_model/runs/pathology/best.pt`.

Each run also writes `run_config.json`, `history.json`, `history.csv`, and
`best_metrics.json` with per-class precision, recall, F1, support, micro-F1,
exact match, and the supported-class macro-F1 used for selection.
