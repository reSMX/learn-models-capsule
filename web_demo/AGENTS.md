# Web MVP

This directory contains the local demonstration UI and API for the two
independent Galar multi-label frame classifiers:

- Anatomy: nine anatomical landmark and GI-section outputs;
- Pathology: twelve pathological finding outputs.

The application is a research demonstration, not a diagnostic system. Keep all
web implementation under `web_demo/` except the dedicated
`requirements-web.txt` and the root README section needed to run it. Do not
change training, datasets, checkpoints, or split policy to support the web app.

## Sources of truth

Read only what is relevant to the current web task:

- local root `AGENTS.md`, when available — repository-wide invariants and
  working-tree safety; this local file is intentionally excluded from Git;
- `galar_dual_model/README.md` — task definitions, training outputs, and known
  limitations;
- `galar_dual_model/labels.py` — exact Anatomy, Pathology, and technical label
  groups;
- `galar_dual_model/train_common.py` — model construction, checkpoint schema,
  validation preprocessing, activation, and fixed-threshold behavior;
- each selected run's `best_metrics.json` and `run_config.json` — authoritative
  model metadata and validation metrics.

`docs/legacy/BEST_MODEL_DIAGNOSTICS.md` and
`runs/combined_galar_head_only/best.pt` describe the older established
14-class softmax classifier. They are not the default model or metric source
for this dual-model web application.

## Runtime checkpoints

Use these defaults relative to the repository root:

- `ANATOMY_MODEL_PATH=galar_dual_model/runs/anatomy_workers0_20260907/best.pt`;
- `PATHOLOGY_MODEL_PATH=galar_dual_model/runs/pathology/best.pt`.

Both paths may be overridden independently through environment variables. Load
the sibling `best_metrics.json` and `run_config.json` for each checkpoint. Model
weights and run artifacts stay outside Git, and the application must not need
the original Galar dataset, metadata, or SSD image cache at runtime.

Fail application startup with a clear owner-facing error if either checkpoint
is absent, incompatible, or mislabeled. Do not silently fall back to the old
14-class checkpoint or operate with only one of the two models.

For every checkpoint:

- require `problem_type == "multi_label"` and `activation == "sigmoid"`;
- require the expected `task` (`anatomy` or `pathology`);
- read class order from `checkpoint["classes"]`; never hardcode output indices;
- reject duplicate classes, an output-size mismatch, or a class set that does
  not match `galar_dual_model/labels.py` after zero-support Pathology columns
  are excluded;
- read the fixed decision threshold from `checkpoint["threshold"]`;
- use only fixed-threshold `best_metrics.json` for model reporting;
- never use `threshold_diagnostics.json` for checkpoint selection or present it
  as independent validation.

## Basic functionality

Build a Russian-language responsive single-page application with FastAPI,
Jinja2, plain HTML/CSS/JavaScript, and no database, authentication, Node.js,
external analytics, CDN, or cloud dependency.

The basic user flow is:

1. choose an image or short video through a file picker or drag-and-drop;
2. preview the selected media locally;
3. submit it for local analysis;
4. run both Anatomy and Pathology models on the same frame or sampled frames;
5. show the two result groups independently, including every fixed-threshold
   positive label, the top three scores, and all label scores;
6. for video, show sampled timestamps and per-frame results without claiming
   temporal event detection;
7. reset the page and analyze another file.

Anatomical and pathological labels may co-occur. Multiple labels inside either
model may also be active. Never convert either output into softmax, force one
winner, suppress valid co-occurrence, or combine the two classifiers into a
single class list.

If no Pathology score crosses its checkpoint threshold, say only that no
finding exceeded the model threshold. Do not call the frame healthy, normal,
or pathology-free. Ranking by score is allowed for display, but a top-ranked
label below threshold is not a positive prediction.

## Expected application structure

```text
web_demo/
  AGENTS.md
  __init__.py
  app.py
  inference.py
  labels.py
  schemas.py
  templates/
    index.html
  static/
    app.js
    styles.css
  tests/
    test_inference.py
    test_api.py
requirements-web.txt
```

Keep the two model wrappers behind one inference service so inputs are decoded
and preprocessed once. Load both checkpoints exactly once during application
startup. Use one bounded inference lock or queue, and run the models
sequentially on the shared batch to avoid unnecessary peak VRAM use.

## Inference contract

Use `DEVICE=auto|cuda|cpu`; `auto` selects CUDA when available and otherwise
uses CPU. The server must work on CPU. Use `torch.inference_mode()`, call
`eval()` on both models, and use CUDA autocast only when supported. Never create
one model copy per request or one CUDA copy per Uvicorn worker.

Construct each network as `torchvision.models.efficientnet_b0(weights=None)`,
replace the final linear layer with the checkpoint class count, and load
`checkpoint["model"]`. Classifier dropout has no effect in evaluation mode, but
when reconstructing it prefer the adjacent `run_config.json` value and fall
back to the training default only when metadata is unavailable.

Use the exact deterministic validation preprocessing from
`galar_dual_model/train_common.py` for both models:

1. decode and apply EXIF orientation;
2. convert to RGB;
3. resize the shorter side to 256;
4. center-crop to `224 x 224`;
5. convert to `float32` in `[0, 1]`;
6. normalize with the mean and standard deviation from
   `EfficientNet_B0_Weights.DEFAULT.transforms()`.

Do not apply training augmentation during inference. Compute independent
sigmoid scores for every output. Return ordinary Python numbers, not Torch or
NumPy scalar objects.

## API contract

Required routes:

- `GET /` — application page;
- `GET /api/health` — readiness, selected device, CUDA availability, and a
  readiness entry for both models;
- `GET /api/models` — task, checkpoint filename, epoch, fixed threshold,
  classes, supported macro-F1, micro-F1, exact match, and per-class metrics for
  each model;
- `POST /api/predict-image` — run both models on one image;
- `POST /api/predict-video` — run both models on uniformly sampled video
  frames.

Image prediction responses must follow one stable shape. For example:

```json
{
  "results": {
    "anatomy": {
      "threshold": 0.5,
      "detected": [],
      "top_predictions": [],
      "all_scores": {}
    },
    "pathology": {
      "threshold": 0.5,
      "detected": [],
      "top_predictions": [],
      "all_scores": {}
    }
  },
  "runtime": {
    "device": "cuda",
    "inference_ms": 12.4
  }
}
```

Each prediction item contains `class_key`, `display_name`, `category`, and
`score`. Sort `detected` and `top_predictions` by descending score. Return a
maximum of three entries in each `top_predictions` list, but return every score
in `all_scores`.

Do not expose absolute local paths in normal API responses. Convert input and
decode errors into stable `4xx` responses; reserve `5xx` for genuine server
failures.

## Image and video handling

For images:

- accept JPEG, PNG, and WebP;
- use Pillow to verify actual content rather than trusting extension or MIME;
- reject empty, corrupt, unsupported, oversized, or decompression-bomb inputs;
- default to a 15 MB encoded-size limit and a 40-megapixel decoded-size limit;
- never persist uploads or log their content or original filename.

For videos:

- support MP4 when the installed decoder can read it reliably;
- enforce configurable encoded-size and duration limits before full analysis;
- sample uniformly at a default of one frame per second and at most 120 frames;
- process frames in configurable batches suitable for an RTX 3060 12 GB;
- create temporary files only in a safe temporary directory and delete them in
  `finally` on success, validation failure, cancellation, and decode failure;
- return duration, sampled-frame count, timestamps, both model results per
  frame, and total processing time;
- return at most 12 reduced JPEG previews for the UI, never base64 for every
  sampled frame;
- do not add hidden temporal smoothing or merge frames into clinical events.

Expose upload limits, video duration, sample rate, maximum sampled frames, and
inference batch size as documented environment settings. Keep defaults visible
near the upload form.

## Labels and presentation

Show a human-readable Russian name first and the exact Galar key as secondary
text. Keep the following categories separate:

- Anatomy landmarks: `z-line`, `pylorus`, `ampulla of vater`,
  `ileocecal valve`;
- Anatomy GI sections: `mouth`, `esophagus`, `stomach`, `small intestine`,
  `colon`;
- Pathology findings: `ulcer`, `polyp`, `active bleeding`, `blood`, `erythema`,
  `erosion`, `angiectasia`, `IBD`, `foreign body`, `hematin`, `cancer`,
  `lymphangioectasis`.

Use this display mapping:

| Source key | Russian display name | Category |
|---|---|---|
| `z-line` | Z-линия | анатомический ориентир |
| `pylorus` | Привратник | анатомический ориентир |
| `ampulla of vater` | Большой дуоденальный сосочек | анатомический ориентир |
| `ileocecal valve` | Илеоцекальный клапан | анатомический ориентир |
| `mouth` | Ротовая полость | отдел ЖКТ |
| `esophagus` | Пищевод | отдел ЖКТ |
| `stomach` | Желудок | отдел ЖКТ |
| `small intestine` | Тонкая кишка | отдел ЖКТ |
| `colon` | Толстая кишка | отдел ЖКТ |
| `ulcer` | Язва | патологическая находка |
| `polyp` | Полип | патологическая находка |
| `active bleeding` | Активное кровотечение | патологическая находка |
| `blood` | Кровь | патологическая находка |
| `erythema` | Эритема | патологическая находка |
| `erosion` | Эрозия | патологическая находка |
| `angiectasia` | Ангиоэктазия | патологическая находка |
| `IBD` | Воспалительное заболевание кишечника (ВЗК) | патологическая находка |
| `foreign body` | Инородное тело | патологическая находка |
| `hematin` | Гематин / изменённая кровь | патологическая находка |
| `cancer` | Злокачественное новообразование | патологическая находка |
| `lymphangioectasis` | Лимфангиоэктазия | патологическая находка |

Use exact source keys in API data. In particular, keep `blood`,
`active bleeding`, and `hematin` separate. Do not rename or map either generic
blood label to hematin. Technical labels are not predictions from either
model. An unknown display key must fall back safely to the source key.

The UI must include:

- an Image/Video switch, drag-and-drop zone, file button, and local preview;
- explicit selected, analyzing, result, and error states;
- separate Anatomy and Pathology panels with threshold-positive results,
  top-three score bars, and expandable all-score lists;
- for video, a compact timeline/table, filters by task and label, and no more
  than 12 key-frame previews;
- a per-model information panel populated from real run artifacts;
- actual CPU/CUDA device and elapsed processing time;
- a reset action, keyboard navigation, visible focus, sufficient contrast, and
  `prefers-reduced-motion` support.

Do not include fabricated patient data, testimonials, diagnoses, nonfunctional
controls, dataset images, or decorative pseudo-medical claims.

## Required safety wording

Keep a visible Russian warning equivalent to:

> Исследовательский прототип. Результаты двух моделей не являются диагнозом и
> не предназначены для принятия медицинских решений без проверки врачом.

Near every result, state that:

- the models analyze isolated frames without temporal or patient context;
- sigmoid scores are uncalibrated model scores, not diagnostic probabilities;
- multiple outputs may be active and some classes perform very poorly;
- a below-threshold Pathology result does not exclude disease;
- uploaded files are processed locally and are not retained.

Never hide zero or weak per-class metrics. Present supported macro-F1 as the
primary aggregate metric for each task; do not let micro-F1 or exact match
visually imply uniformly good class performance.

## Security and runtime limits

- Listen on `127.0.0.1` by default.
- Do not enable permissive CORS, telemetry, or external uploads.
- Never construct a filesystem path from an uploaded filename.
- Bound upload size, decoded pixels, video duration, sampled frames, batch
  size, and concurrent inference.
- Use one Uvicorn worker when CUDA is enabled.
- Clean all temporary files even when a request fails or is cancelled.
- Return a clear busy response instead of exhausting RAM or VRAM.

## Validation

At minimum, automate these checks:

1. each real checkpoint loads on CPU with the correct task and class set;
2. Anatomy returns exactly nine finite sigmoid scores in `[0, 1]`;
3. Pathology returns exactly twelve finite sigmoid scores in `[0, 1]`;
4. sigmoid scores are not required to sum to one, and multiple labels can cross
   the fixed threshold;
5. preprocessing returns a `3 x 224 x 224` tensor;
6. one image response contains both independent result groups and sorted top
   threes;
7. task mismatch, missing checkpoint, and incompatible class order fail with a
   clear startup error;
8. valid PNG/JPEG/WebP pass while disguised text, empty, corrupt, oversized,
   and excessive-pixel inputs are rejected with `4xx`;
9. a short generated test MP4 passes and its temporary file is removed;
10. health and models endpoints match the documented schema;
11. API unit tests run without CUDA through mock inference where real weights
    are unnecessary;
12. CPU smoke inference passes for both real checkpoints; when CUDA is
    available, run the same smoke check on CUDA.

Do not run training as validation. Record measured local image/video timing
without promising the same performance on other hardware.

## Documentation and completion

Keep web dependencies in `requirements-web.txt`. Document in the root README:

```powershell
.\.venv-win\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv-win\Scripts\python.exe -m uvicorn web_demo.app:app --host 127.0.0.1 --port 8000
```

Document `ANATOMY_MODEL_PATH`, `PATHOLOGY_MODEL_PATH`, `DEVICE`, upload/video
limits, sampling settings, batch size, and common errors. The finished MVP must
start without either dataset, load each model once, analyze images and short
MP4 files with both models, expose honest per-model metrics, handle invalid
files without crashing, and leave no uploaded or temporary files behind.

Before finishing a web task, run the narrowest relevant tests, inspect
`git status`, and report changed files, checks performed, and known limitations.
Do not stage or alter the protected dirty artifacts under
`runs/regularized_v2`.
