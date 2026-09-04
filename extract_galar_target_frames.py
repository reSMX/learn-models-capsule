"""Import clean target frames from Galar videos or official PNG folders.

This script is intentionally separate from training.  It uses the video list
created by ``select_galar_target_videos.py``, reads the corresponding Galar
``Labels/*.csv`` files, and extracts only frames for these mappings:

* ``ampulla of vater`` -> ``Ampulla of Vater``
* ``hematin`` -> ``Blood - hematin``
* ``polyp`` -> ``Polyp``

Frames carrying another pathological label are audited and excluded from the
current single-label softmax task.  GI-section and anatomical landmark labels
do not count as conflicts.

The script can decode each video sequentially through ffmpeg, or copy the exact
requested frames from the official ``<study_id>/frame_XXXXXX.PNG`` folders.
No OpenCV dependency is required.

Example to run after the selected videos have been downloaded::

    python -B extract_galar_target_frames.py --videos "D:\\Galar\\Videos"
    python -B extract_galar_target_frames.py --frames-root "D:\\Dataset_galar" \
        --galar-root "D:\\Dataset_galar" --skip-missing-studies

Use ``--video-map`` if the downloaded filenames are not ``1.mp4``, ``2.mp4``,
etc.  The mapping CSV must contain ``study_id`` and ``video_path`` columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from select_galar_target_videos import (
    OTHER_PATHOLOGY_COLUMNS,
    TARGET_COLUMNS,
    is_positive,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg"}

FINDING_CLASSES = {
    "ampulla_of_vater": "Ampulla of Vater",
    "blood_hematin": "Blood - hematin",
    "polyp": "Polyp",
}

FINDING_CATEGORIES = {
    "ampulla_of_vater": "Anatomy",
    "blood_hematin": "Luminal",
    "polyp": "Luminal",
}

METADATA_COLUMNS = (
    "filename",
    "video_id",
    "frame_number",
    "finding_category",
    "finding_class",
    "x1",
    "y1",
    "x2",
    "y2",
    "x3",
    "y3",
    "x4",
    "y4",
)

SOURCE_COLUMNS = (
    "source_dataset",
    "source_study_id",
    "source_frame_number",
    "capsule_system",
    "gender",
    "age",
)


@dataclass(frozen=True)
class Study:
    study_id: str
    video_id: str
    label_file: Path
    capsule_system: str
    gender: str
    age: str
    manifest_row: dict[str, str]


@dataclass(frozen=True)
class FrameRecord:
    study_id: str
    video_id: str
    frame_number: int
    target: str
    capsule_system: str
    gender: str
    age: str

    @property
    def filename(self) -> str:
        return f"{self.video_id}_{self.frame_number:08d}.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import clean ampulla/hematin/polyp frames from selected Galar studies."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--videos",
        type=Path,
        help="Directory containing the downloaded Galar video files",
    )
    source.add_argument(
        "--frames-root",
        type=Path,
        help=(
            "Directory containing official pre-extracted study folders such as "
            "5/frame_003735.PNG"
        ),
    )
    parser.add_argument(
        "--galar-root",
        type=Path,
        default=Path("another db"),
        help="Directory containing metadata.csv and Labels/ (default: another db)",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("another db/galar_target_videos.csv"),
        help="Video manifest created by select_galar_target_videos.py",
    )
    parser.add_argument(
        "--video-map",
        type=Path,
        help="Optional CSV with study_id,video_path for nonstandard video filenames",
    )
    parser.add_argument(
        "--output-images",
        type=Path,
        default=Path(r"D:\dataset_quazir\galar_target_images"),
        help="Destination for extracted class folders",
    )
    parser.add_argument(
        "--galar-metadata-output",
        type=Path,
        default=Path(r"D:\dataset_quazir\galar_target_metadata.csv"),
        help="Galar-only train.py-compatible metadata CSV",
    )
    parser.add_argument(
        "--base-metadata",
        type=Path,
        default=Path(r"D:\dataset_quazir\metadata.csv"),
        help="Existing Kvasir metadata used to create a combined CSV",
    )
    parser.add_argument(
        "--combined-metadata-output",
        type=Path,
        default=Path(r"D:\dataset_quazir\metadata_with_galar.csv"),
        help="New combined Kvasir + Galar metadata CSV",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path(r"D:\dataset_quazir\galar_excluded_frames.csv"),
        help="Audit CSV for conflicting or extraction-capped target frames",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(r"D:\dataset_quazir\galar_extraction_summary.json"),
        help="JSON extraction summary",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable name or path (default: ffmpeg)",
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="ffprobe executable name or path (default: ffprobe)",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Offset added only when locating a decoded video frame (default: 0)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100 (default: 95)",
    )
    parser.add_argument(
        "--max-frames-per-video-class",
        type=int,
        default=0,
        help=(
            "Optional even temporal cap during import; 0 keeps all clean target frames. "
            "The training pipeline already caps training groups at 500."
        ),
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Replace already imported target images instead of reusing them",
    )
    parser.add_argument(
        "--replace-metadata",
        action="store_true",
        help="Allow replacement of this script's metadata/audit/summary outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve source frames and report selections without copying/decoding or writing",
    )
    parser.add_argument(
        "--skip-missing-studies",
        action="store_true",
        help=(
            "Omit manifest studies whose source folder is unavailable and record them in "
            "the summary; intended for incomplete official downloads"
        ),
    )
    args = parser.parse_args()

    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.max_frames_per_video_class < 0:
        parser.error("--max-frames-per-video-class cannot be negative")
    return args


def read_csv(path: Path, delimiter: str = ",") -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def safe_child(root: Path, relative_path: Path) -> Path:
    root_resolved = root.resolve()
    result = (root / relative_path).resolve()
    if result != root_resolved and root_resolved not in result.parents:
        raise ValueError(f"Path escapes Galar root: {relative_path}")
    return result


def load_studies(manifest_path: Path, galar_root: Path) -> list[Study]:
    _, rows = read_csv(manifest_path)
    required = {
        "source_dataset",
        "study_id",
        "video_id",
        "label_file",
        "capsule_system",
        "gender",
        "age",
    }
    if not rows:
        raise ValueError(f"Selection manifest is empty: {manifest_path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Selection manifest is missing columns: {sorted(missing)}")

    studies: list[Study] = []
    seen_ids: set[str] = set()
    for row in rows:
        study_id = row["study_id"].strip()
        if row["source_dataset"].strip() != "Galar":
            raise ValueError(f"Unexpected source_dataset for study {study_id}")
        if study_id in seen_ids:
            raise ValueError(f"Duplicate study_id in selection manifest: {study_id}")
        seen_ids.add(study_id)
        label_file = safe_child(galar_root, Path(row["label_file"]))
        if not label_file.is_file():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        studies.append(
            Study(
                study_id=study_id,
                video_id=row["video_id"].strip(),
                label_file=label_file,
                capsule_system=row["capsule_system"].strip(),
                gender=row["gender"].strip(),
                age=row["age"].strip(),
                manifest_row=row,
            )
        )
    return sorted(studies, key=lambda study: int(study.study_id))


def filter_available_frame_studies(
    studies: list[Study], frames_root: Path, skip_missing: bool
) -> tuple[list[Study], list[Study]]:
    if not frames_root.is_dir():
        raise FileNotFoundError(f"Pre-extracted Galar frame directory not found: {frames_root}")
    available: list[Study] = []
    missing: list[Study] = []
    for study in studies:
        if (frames_root / study.study_id).is_dir():
            available.append(study)
        else:
            missing.append(study)
    if missing and not skip_missing:
        missing_ids = ", ".join(study.study_id for study in missing)
        raise FileNotFoundError(
            f"Source frame folders are missing for studies: {missing_ids}. "
            "Use --skip-missing-studies only when the incomplete download is intentional."
        )
    if not available:
        raise ValueError("None of the selected Galar studies has a source frame folder")
    return available, missing


def audit_row(
    study: Study,
    frame_number: int,
    active_targets: Iterable[str],
    other_pathologies: Iterable[str],
    reason: str,
) -> dict[str, object]:
    return {
        "source_dataset": "Galar",
        "study_id": study.study_id,
        "video_id": study.video_id,
        "frame_number": frame_number,
        "active_target_labels": "|".join(sorted(active_targets)),
        "other_pathology_labels": "|".join(sorted(other_pathologies)),
        "reason": reason,
    }


def load_clean_frames(studies: list[Study]) -> tuple[list[FrameRecord], list[dict[str, object]]]:
    records: list[FrameRecord] = []
    audit: list[dict[str, object]] = []

    for study in studies:
        with study.label_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = set(TARGET_COLUMNS) | set(OTHER_PATHOLOGY_COLUMNS) | {"frame"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{study.label_file}: missing columns: {sorted(missing)}")

            clean_by_frame: dict[int, list[str]] = defaultdict(list)
            for row in reader:
                active_targets = [
                    target
                    for source, target in TARGET_COLUMNS.items()
                    if is_positive(row[source])
                ]
                if not active_targets:
                    continue
                frame_number = int(row["frame"])
                other_pathologies = [
                    column for column in OTHER_PATHOLOGY_COLUMNS if is_positive(row[column])
                ]
                if len(active_targets) != 1:
                    audit.append(
                        audit_row(
                            study,
                            frame_number,
                            active_targets,
                            other_pathologies,
                            "multiple_requested_targets",
                        )
                    )
                    continue
                if other_pathologies:
                    audit.append(
                        audit_row(
                            study,
                            frame_number,
                            active_targets,
                            other_pathologies,
                            "other_pathology_present",
                        )
                    )
                    continue
                clean_by_frame[frame_number].append(active_targets[0])

        for frame_number, targets in sorted(clean_by_frame.items()):
            unique_targets = sorted(set(targets))
            if len(unique_targets) != 1:
                audit.append(
                    audit_row(
                        study,
                        frame_number,
                        unique_targets,
                        (),
                        "conflicting_duplicate_rows",
                    )
                )
                continue
            records.append(
                FrameRecord(
                    study_id=study.study_id,
                    video_id=study.video_id,
                    frame_number=frame_number,
                    target=unique_targets[0],
                    capsule_system=study.capsule_system,
                    gender=study.gender,
                    age=study.age,
                )
            )

    return records, audit


def validate_manifest_counts(studies: list[Study], records: list[FrameRecord]) -> None:
    counts: Counter[tuple[str, str]] = Counter(
        (record.study_id, record.target) for record in records
    )
    for study in studies:
        for target in TARGET_COLUMNS.values():
            field = f"{target}_usable_frames"
            expected = int(study.manifest_row[field])
            actual = counts[(study.study_id, target)]
            if actual != expected:
                raise ValueError(
                    f"Selection manifest drift for study {study.study_id}, {target}: "
                    f"manifest={expected}, labels={actual}. Re-run select_galar_target_videos.py."
                )


def apply_temporal_cap(
    records: list[FrameRecord],
    maximum: int,
    studies_by_id: dict[str, Study],
) -> tuple[list[FrameRecord], list[dict[str, object]]]:
    if maximum == 0:
        return records, []

    grouped: dict[tuple[str, str], list[FrameRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.study_id, record.target)].append(record)

    kept: list[FrameRecord] = []
    audit: list[dict[str, object]] = []
    for (study_id, target), group in sorted(
        grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])
    ):
        group.sort(key=lambda record: record.frame_number)
        if len(group) <= maximum:
            kept.extend(group)
            continue
        if maximum == 1:
            keep_positions = {0}
        else:
            keep_positions = {
                index * (len(group) - 1) // (maximum - 1) for index in range(maximum)
            }
        study = studies_by_id[study_id]
        for index, record in enumerate(group):
            if index in keep_positions:
                kept.append(record)
            else:
                audit.append(
                    audit_row(
                        study,
                        record.frame_number,
                        (target,),
                        (),
                        "extraction_temporal_cap",
                    )
                )
    kept.sort(key=lambda record: (int(record.study_id), record.frame_number, record.target))
    return kept, audit


def load_video_map(path: Path, videos_root: Path) -> dict[str, Path]:
    _, rows = read_csv(path)
    required = {"study_id", "video_path"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"Video map must contain columns: {sorted(required)}")
    mapping: dict[str, Path] = {}
    for row in rows:
        study_id = row["study_id"].strip()
        video_path = Path(row["video_path"].strip())
        if not video_path.is_absolute():
            video_path = videos_root / video_path
        video_path = video_path.resolve()
        if study_id in mapping:
            raise ValueError(f"Duplicate study_id in video map: {study_id}")
        mapping[study_id] = video_path
    return mapping


def resolve_videos(
    studies: list[Study], videos_root: Path, video_map_path: Path | None
) -> dict[str, Path]:
    if not videos_root.is_dir():
        raise FileNotFoundError(f"Video directory not found: {videos_root}")

    explicit = load_video_map(video_map_path, videos_root) if video_map_path else {}
    scanned = [
        path.resolve()
        for path in videos_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in scanned:
        by_stem[path.stem.casefold()].append(path)

    resolved: dict[str, Path] = {}
    for study in studies:
        if study.study_id in explicit:
            candidates = [explicit[study.study_id]]
        else:
            aliases = {study.study_id.casefold(), study.video_id.casefold()}
            candidates = sorted(
                {candidate for alias in aliases for candidate in by_stem.get(alias, [])}
            )
        if not candidates:
            raise FileNotFoundError(
                f"No video found for study {study.study_id}. Expected a stem equal to "
                f"{study.study_id!r} or {study.video_id!r}; otherwise use --video-map."
            )
        if len(candidates) != 1:
            raise ValueError(
                f"Multiple video candidates for study {study.study_id}: "
                + ", ".join(str(path) for path in candidates)
            )
        video_path = candidates[0]
        if not video_path.is_file():
            raise FileNotFoundError(f"Mapped video not found: {video_path}")
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {video_path}")
        resolved[study.study_id] = video_path

    unknown_mapped = sorted(set(explicit) - {study.study_id for study in studies}, key=int)
    if unknown_mapped:
        print("Warning: unused study IDs in --video-map:", ", ".join(unknown_mapped))
    return resolved


def find_executable(command: str) -> str:
    direct = Path(command)
    if direct.is_file():
        return str(direct.resolve())
    found = shutil.which(command)
    if found is None:
        raise FileNotFoundError(
            f"Executable not found: {command}. Install ffmpeg or pass its path explicitly."
        )
    return found


def probe_dimensions(ffprobe: str, video_path: Path) -> tuple[int, int]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Could not identify one video stream in {video_path}")
    width = int(streams[0]["width"])
    height = int(streams[0]["height"])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions for {video_path}: {width}x{height}")
    return width, height


def read_exact(stream, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def output_filename(record: FrameRecord, suffix: str = ".jpg") -> str:
    return f"{record.video_id}_{record.frame_number:08d}{suffix.lower()}"


def image_path(
    output_root: Path, record: FrameRecord, suffix: str = ".jpg"
) -> Path:
    return output_root / record.target / output_filename(record, suffix)


def source_frame_path(frames_root: Path, record: FrameRecord) -> Path:
    return frames_root / record.study_id / f"frame_{record.frame_number:06d}.PNG"


def validate_existing_image(path: Path) -> None:
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Existing extracted image is empty: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:
        raise RuntimeError(f"Existing extracted image is invalid: {path}") from error


def resolve_preextracted_frames(
    frames_root: Path, records: list[FrameRecord]
) -> dict[FrameRecord, Path]:
    if not frames_root.is_dir():
        raise FileNotFoundError(f"Pre-extracted Galar frame directory not found: {frames_root}")
    resolved: dict[FrameRecord, Path] = {}
    missing: list[Path] = []
    for record in records:
        source = source_frame_path(frames_root, record)
        if source.is_file():
            resolved[record] = source
        else:
            missing.append(source)
    if missing:
        preview = ", ".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} annotated Galar source frames are missing ({preview})"
        )
    return resolved


def copy_preextracted_frames(
    sources: dict[FrameRecord, Path],
    records: list[FrameRecord],
    output_root: Path,
    overwrite_images: bool,
) -> tuple[int, int]:
    copied = 0
    reused = 0
    for record in records:
        source = sources[record]
        target = image_path(output_root, record, ".png")
        if target.exists() and not overwrite_images:
            validate_existing_image(target)
            reused += 1
            continue
        validate_existing_image(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".part.png")
        shutil.copy2(source, temporary)
        validate_existing_image(temporary)
        temporary.replace(target)
        copied += 1
    return copied, reused


def save_jpeg(payload: bytes, width: int, height: int, target: Path, quality: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part.jpg")
    image = Image.frombytes("RGB", (width, height), payload)
    image.save(temporary, format="JPEG", quality=quality, optimize=True)
    temporary.replace(target)


def extract_video_frames(
    ffmpeg: str,
    ffprobe: str,
    video_path: Path,
    records: list[FrameRecord],
    output_root: Path,
    frame_offset: int,
    jpeg_quality: int,
    overwrite_images: bool,
) -> tuple[int, int]:
    requested: dict[int, FrameRecord] = {}
    reused = 0
    for record in records:
        decoded_index = record.frame_number + frame_offset
        if decoded_index < 0:
            raise ValueError(
                f"Negative decoded frame index for {record.video_id}, frame {record.frame_number}"
            )
        if decoded_index in requested:
            raise ValueError(
                f"Multiple clean labels resolve to decoded frame {decoded_index} in {record.video_id}"
            )
        target = image_path(output_root, record)
        if target.exists() and not overwrite_images:
            validate_existing_image(target)
            reused += 1
        else:
            requested[decoded_index] = record

    if not requested:
        return 0, reused

    width, height = probe_dimensions(ffprobe, video_path)
    bytes_per_frame = width * height * 3
    maximum_index = max(requested)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-noautorotate",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Could not open ffmpeg pipes")

    extracted = 0
    decoded_index = 0
    try:
        while decoded_index <= maximum_index:
            payload = read_exact(process.stdout, bytes_per_frame)
            if len(payload) != bytes_per_frame:
                break
            record = requested.get(decoded_index)
            if record is not None:
                save_jpeg(
                    payload,
                    width,
                    height,
                    image_path(output_root, record),
                    jpeg_quality,
                )
                extracted += 1
            decoded_index += 1
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stderr = process.stderr.read()
        process.stderr.close()

    missing_indices = sorted(set(requested) - set(range(decoded_index)))
    if missing_indices:
        error_text = stderr.decode(errors="replace").strip()
        preview = ", ".join(str(index) for index in missing_indices[:10])
        raise RuntimeError(
            f"Video {video_path} ended before {len(missing_indices)} requested frames "
            f"({preview}). Check --frame-offset and the source video. ffmpeg: {error_text}"
        )
    return extracted, reused


def galar_metadata_row(
    record: FrameRecord, suffix: str = ".jpg"
) -> dict[str, object]:
    row: dict[str, object] = {
        "filename": output_filename(record, suffix),
        "video_id": record.video_id,
        "frame_number": record.frame_number,
        "finding_category": FINDING_CATEGORIES[record.target],
        "finding_class": FINDING_CLASSES[record.target],
        "source_dataset": "Galar",
        "source_study_id": record.study_id,
        "source_frame_number": record.frame_number,
        "capsule_system": record.capsule_system,
        "gender": record.gender,
        "age": record.age,
    }
    for coordinate in METADATA_COLUMNS[5:]:
        row[coordinate] = ""
    return row


def ensure_replaceable(paths: Iterable[Path], replace: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not replace:
        raise FileExistsError(
            "Output files already exist; use --replace-metadata to replace them: "
            + ", ".join(str(path) for path in existing)
        )


def write_csv_atomic(
    path: Path,
    rows: Iterable[dict[str, object]],
    fieldnames: list[str],
    delimiter: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter=delimiter,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_combined_rows(
    base_metadata: Path,
    galar_rows: list[dict[str, object]],
) -> tuple[list[str], list[dict[str, object]]]:
    base_fields, base_rows = read_csv(base_metadata, delimiter=";")
    missing = set(METADATA_COLUMNS) - set(base_fields)
    if missing:
        raise ValueError(f"Base Kvasir metadata is missing columns: {sorted(missing)}")

    fieldnames = list(base_fields)
    for column in SOURCE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    combined: list[dict[str, object]] = []
    known_filenames: set[str] = set()
    for row in base_rows:
        filename = row["filename"]
        if filename in known_filenames:
            # Preserve the existing Kvasir duplicate/multi-label audit behavior.
            pass
        known_filenames.add(filename)
        augmented: dict[str, object] = dict(row)
        augmented["source_dataset"] = augmented.get("source_dataset") or "Kvasir-Capsule"
        augmented["source_study_id"] = augmented.get("source_study_id") or row["video_id"]
        augmented["source_frame_number"] = (
            augmented.get("source_frame_number") or row["frame_number"]
        )
        augmented.setdefault("capsule_system", "")
        augmented.setdefault("gender", "")
        augmented.setdefault("age", "")
        combined.append(augmented)

    for row in galar_rows:
        filename = str(row["filename"])
        if filename in known_filenames:
            raise ValueError(f"Galar filename collides with existing metadata: {filename}")
        known_filenames.add(filename)
        combined.append(row)
    return fieldnames, combined


def class_counts(records: list[FrameRecord]) -> dict[str, int]:
    counts = Counter(record.target for record in records)
    return {target: counts[target] for target in TARGET_COLUMNS.values()}


def main() -> None:
    args = parse_args()
    manifest_studies = load_studies(args.selection_manifest, args.galar_root)
    missing_studies: list[Study] = []
    if args.frames_root is not None:
        studies, missing_studies = filter_available_frame_studies(
            manifest_studies, args.frames_root, args.skip_missing_studies
        )
    else:
        studies = manifest_studies
    records, audit = load_clean_frames(studies)
    validate_manifest_counts(studies, records)
    records, cap_audit = apply_temporal_cap(
        records,
        args.max_frames_per_video_class,
        {study.study_id: study for study in studies},
    )
    audit.extend(cap_audit)
    resolved_videos: dict[str, Path] = {}
    frame_sources: dict[FrameRecord, Path] = {}
    output_suffix = ".jpg"
    if args.frames_root is not None:
        frame_sources = resolve_preextracted_frames(args.frames_root, records)
        output_suffix = ".png"
    else:
        resolved_videos = resolve_videos(studies, args.videos, args.video_map)

    print(f"Manifest studies: {len(manifest_studies)}")
    print(f"Available selected studies: {len(studies)}")
    if missing_studies:
        print(
            "Intentionally omitted missing studies: "
            + ", ".join(study.study_id for study in missing_studies)
        )
    print(f"Clean target frames: {len(records)}")
    print("Class counts:", json.dumps(class_counts(records), ensure_ascii=False))
    print(f"Audited exclusions: {len(audit)}")
    if args.dry_run:
        print("Dry run: all requested source frames resolved; no files written.")
        return

    metadata_outputs = (
        args.galar_metadata_output,
        args.combined_metadata_output,
        args.audit_output,
        args.summary_output,
    )
    ensure_replaceable(metadata_outputs, args.replace_metadata)
    ffmpeg = find_executable(args.ffmpeg) if args.frames_root is None else ""
    ffprobe = find_executable(args.ffprobe) if args.frames_root is None else ""

    records_by_study: dict[str, list[FrameRecord]] = defaultdict(list)
    for record in records:
        records_by_study[record.study_id].append(record)

    extracted_total = 0
    reused_total = 0
    for index, study in enumerate(studies, 1):
        study_records = records_by_study[study.study_id]
        if args.frames_root is not None:
            print(
                f"[{index}/{len(studies)}] study {study.study_id}: "
                f"copying {len(study_records)} exact PNG frames"
            )
            extracted, reused = copy_preextracted_frames(
                sources=frame_sources,
                records=study_records,
                output_root=args.output_images,
                overwrite_images=args.overwrite_images,
            )
        else:
            print(
                f"[{index}/{len(studies)}] study {study.study_id}: "
                f"{len(study_records)} frames from {resolved_videos[study.study_id]}"
            )
            extracted, reused = extract_video_frames(
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                video_path=resolved_videos[study.study_id],
                records=study_records,
                output_root=args.output_images,
                frame_offset=args.frame_offset,
                jpeg_quality=args.jpeg_quality,
                overwrite_images=args.overwrite_images,
            )
        extracted_total += extracted
        reused_total += reused

    missing_images = [
        image_path(args.output_images, record, output_suffix)
        for record in records
        if not image_path(args.output_images, record, output_suffix).is_file()
    ]
    if missing_images:
        raise RuntimeError(f"Extraction finished with {len(missing_images)} missing images")

    galar_rows = [galar_metadata_row(record, output_suffix) for record in records]
    galar_fields = list(METADATA_COLUMNS) + list(SOURCE_COLUMNS)
    combined_fields, combined_rows = build_combined_rows(args.base_metadata, galar_rows)

    write_csv_atomic(args.galar_metadata_output, galar_rows, galar_fields, ";")
    write_csv_atomic(args.combined_metadata_output, combined_rows, combined_fields, ";")
    audit_fields = [
        "source_dataset",
        "study_id",
        "video_id",
        "frame_number",
        "active_target_labels",
        "other_pathology_labels",
        "reason",
    ]
    write_csv_atomic(args.audit_output, audit, audit_fields, ";")

    summary: dict[str, object] = {
        "source_dataset": "Galar",
        "source_mode": "pre_extracted_png" if args.frames_root is not None else "video",
        "manifest_selected_studies": len(manifest_studies),
        "selected_studies": len(studies),
        "selected_video_ids": [study.video_id for study in studies],
        "omitted_missing_studies": [study.study_id for study in missing_studies],
        "omitted_missing_class_counts": {
            target: sum(
                int(study.manifest_row[f"{target}_usable_frames"])
                for study in missing_studies
            )
            for target in TARGET_COLUMNS.values()
        },
        "clean_target_frames": len(records),
        "class_counts": class_counts(records),
        "audited_exclusions": len(audit),
        "exclusion_reasons": dict(Counter(str(row["reason"]) for row in audit)),
        "extracted_images": extracted_total,
        "reused_images": reused_total,
        "frame_offset": args.frame_offset,
        "max_frames_per_video_class": args.max_frames_per_video_class,
        "output_images": str(args.output_images.resolve()),
        "galar_metadata": str(args.galar_metadata_output.resolve()),
        "combined_metadata": str(args.combined_metadata_output.resolve()),
        "frames_root": str(args.frames_root.resolve()) if args.frames_root else None,
        "video_files": {
            f"galar_{study_id}": str(path) for study_id, path in resolved_videos.items()
        },
        "split_policy": "Split only by video_id; never split neighboring frames randomly.",
        "model_selection_policy": "Do not use diagnostic-only results for checkpoint selection.",
    }
    write_json_atomic(args.summary_output, summary)

    print(f"Extracted images: {extracted_total}; reused images: {reused_total}")
    print(f"Galar metadata: {args.galar_metadata_output}")
    print(f"Combined metadata: {args.combined_metadata_output}")
    print(
        "For training, point --images at D:\\dataset_quazir (so both Kvasir and Galar "
        "class folders are indexed) and --metadata at the combined CSV."
    )


if __name__ == "__main__":
    main()
