"""Build a compact, seekable SSD image cache for the thinned Galar tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm


CACHE_FORMAT = "galar_jpeg_shards_v1"
INDEX_RECORD = struct.Struct("<QI")
METADATA_FIELDS = ("image_path", "study_id", "video_id", "frame_number", "labels")
METADATA_NAMES = (
    "anatomy_train.csv",
    "anatomy_validation.csv",
    "pathology_train.csv",
    "pathology_validation.csv",
)
SUBSAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize the union of generated Anatomy/Pathology metadata frames and "
            "pack them into seekable per-study JPEG shards for SSD training."
        )
    )
    parser.add_argument("--galar-root", type=Path, required=True)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "metadata",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--jpeg-subsampling",
        choices=tuple(SUBSAMPLING),
        default="4:4:4",
        help="4:4:4 preserves subtle colour detail and is recommended for pathology",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Image decode/resize threads used while each study is packed",
    )
    parser.add_argument(
        "--conversion-batch-size",
        type=int,
        default=64,
        help="Bounded number of converted JPEGs waiting to be written",
    )
    parser.add_argument(
        "--minimum-free-gb",
        type=float,
        default=15.0,
        help="Abort before free space on the destination volume falls below this many GiB",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted cache build with identical settings and metadata",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan metadata and convert a small sample in memory without writing a cache",
    )
    parser.add_argument("--dry-run-samples", type=int, default=32)
    args = parser.parse_args()
    if args.image_size < 256:
        parser.error("image-size must be at least 256 for the current 224-pixel pipeline")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality must be between 1 and 100")
    if args.workers < 1 or args.conversion_batch_size < 1:
        parser.error("workers and conversion-batch-size must be positive")
    if args.minimum_free_gb < 0 or args.dry_run_samples < 1:
        parser.error("minimum-free-gb cannot be negative; dry-run-samples must be positive")
    if args.resume and args.dry_run:
        parser.error("--resume and --dry-run cannot be used together")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_required_frames(
    metadata_dir: Path,
) -> tuple[dict[str, dict[int, str]], dict[str, dict[str, object]]]:
    """Read the union of both tasks without duplicating shared frames."""

    studies: dict[str, dict[int, str]] = {}
    fingerprints: dict[str, dict[str, object]] = {}
    for name in METADATA_NAMES:
        path = metadata_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Generated metadata is missing: {path}. Run prepare_metadata.py first."
            )
        row_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != METADATA_FIELDS:
                raise ValueError(
                    f"Unexpected metadata columns in {path}: {reader.fieldnames}; "
                    f"expected {METADATA_FIELDS}"
                )
            for row_number, row in enumerate(reader, start=2):
                study_id = row["study_id"]
                if not study_id.isdigit():
                    raise ValueError(f"Invalid study_id at {path}:{row_number}: {study_id!r}")
                try:
                    frame_number = int(row["frame_number"])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid frame_number at {path}:{row_number}"
                    ) from error
                relative = Path(row["image_path"])
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0] != study_id
                ):
                    raise ValueError(
                        f"Unsafe or inconsistent image_path at {path}:{row_number}: {relative}"
                    )
                frames = studies.setdefault(study_id, {})
                previous = frames.setdefault(frame_number, relative.as_posix())
                if previous != relative.as_posix():
                    raise ValueError(
                        f"Conflicting paths for study {study_id}, frame {frame_number}: "
                        f"{previous!r} and {relative.as_posix()!r}"
                    )
                row_count += 1
        fingerprints[name] = {
            "bytes": path.stat().st_size,
            "rows": row_count,
            "sha256": sha256_file(path),
        }
    if not studies:
        raise ValueError(f"No frames found in metadata directory: {metadata_dir}")
    return studies, fingerprints


def convert_image(
    source: Path, image_size: int, quality: int, subsampling: int
) -> bytes:
    try:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            if image.size != (image_size, image_size):
                image = image.resize(
                    (image_size, image_size), resample=Image.Resampling.LANCZOS
                )
            output = io.BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=quality,
                subsampling=subsampling,
                optimize=False,
                progressive=False,
            )
            return output.getvalue()
    except Exception as error:
        raise RuntimeError(f"Could not convert Galar frame: {source}") from error


def chunks(values: list[tuple[int, str]], size: int) -> Iterable[list[tuple[int, str]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def existing_volume_path(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"No existing parent volume for output path: {path}")
    return candidate


def free_gib(path: Path) -> float:
    return shutil.disk_usage(existing_volume_path(path)).free / 1024**3


def ensure_space(
    path: Path, minimum_free_gib: float, pending_write_bytes: int = 0
) -> None:
    available_bytes = shutil.disk_usage(existing_volume_path(path)).free
    minimum_bytes = int(minimum_free_gib * 1024**3)
    if available_bytes - pending_write_bytes < minimum_bytes:
        raise RuntimeError(
            f"Cache build stopped before the next write: destination has "
            f"{available_bytes / 1024**3:.2f} GiB free and the write needs "
            f"{pending_write_bytes / 1024**2:.1f} MiB, which would cross the "
            f"required {minimum_free_gib:.2f} GiB reserve. Partial study files "
            "were not promoted and the build can be resumed."
        )


def build_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "format": CACHE_FORMAT,
        "image_size": args.image_size,
        "jpeg_quality": args.jpeg_quality,
        "jpeg_subsampling": args.jpeg_subsampling,
        "index_record": "little-endian uint64 offset + uint32 length",
        "index_record_bytes": INDEX_RECORD.size,
        "minimum_free_gib": args.minimum_free_gb,
    }


def dry_run(
    args: argparse.Namespace,
    galar_root: Path,
    studies: dict[str, dict[int, str]],
) -> None:
    total = sum(len(frames) for frames in studies.values())
    sample_count = min(args.dry_run_samples, total)
    selected_positions = {
        round(index * (total - 1) / max(1, sample_count - 1))
        for index in range(sample_count)
    }
    selected: list[Path] = []
    position = 0
    for study_id in sorted(studies, key=int):
        for frame_number, relative in sorted(studies[study_id].items()):
            del frame_number
            if position in selected_positions:
                selected.append(galar_root / Path(relative))
            position += 1
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        encoded = list(
            pool.map(
                lambda path: convert_image(
                    path,
                    args.image_size,
                    args.jpeg_quality,
                    SUBSAMPLING[args.jpeg_subsampling],
                ),
                selected,
            )
        )
    mean_bytes = sum(map(len, encoded)) / max(1, len(encoded))
    estimated = mean_bytes * total
    print(
        f"Dry run passed: studies={len(studies)}, unique frames={total:,}, "
        f"sample={len(encoded)}, mean JPEG={mean_bytes / 1024:.1f} KiB"
    )
    print(
        f"Estimated cache payload={estimated / 1024**3:.2f} GiB; "
        f"destination free now={free_gib(args.output_root.resolve()):.2f} GiB; "
        "no files written."
    )


def load_or_create_state(
    args: argparse.Namespace,
    output_root: Path,
    metadata_fingerprints: dict[str, dict[str, object]],
) -> dict[str, object]:
    state_path = output_root / "build_state.json"
    settings = build_settings(args)
    expected = {
        "status": "building",
        "settings": settings,
        "metadata_files": metadata_fingerprints,
        "completed_studies": {},
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("settings") == settings
            and manifest.get("metadata_files") == metadata_fingerprints
        ):
            print(f"Cache is already complete: {output_root}")
            return {"already_complete": True}
        raise FileExistsError(
            f"A different completed cache already exists: {output_root}. "
            "Choose another --output-root."
        )
    if output_root.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output directory already exists without a complete manifest: {output_root}. "
                "Use --resume only if it is an interrupted build from this script."
            )
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Cannot safely resume because build_state.json is missing: {output_root}"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("settings") != settings:
            raise ValueError("Resume settings differ from the interrupted cache build")
        if state.get("metadata_files") != metadata_fingerprints:
            raise ValueError(
                "Generated metadata changed after the interrupted cache build; "
                "use a new --output-root"
            )
        return state

    output_root.mkdir(parents=True)
    (output_root / "shards").mkdir()
    (output_root / "indexes").mkdir()
    write_json_atomic(state_path, expected)
    return expected


def main() -> None:
    args = parse_args()
    galar_root = args.galar_root.resolve()
    metadata_dir = args.metadata_dir.resolve()
    output_root = args.output_root.resolve()
    if not galar_root.is_dir():
        raise FileNotFoundError(f"Galar image root not found: {galar_root}")
    studies, metadata_fingerprints = read_required_frames(metadata_dir)
    total_frames = sum(len(frames) for frames in studies.values())
    print(
        f"Metadata union: {total_frames:,} unique frames from {len(studies)} studies "
        f"({sum(item['rows'] for item in metadata_fingerprints.values()):,} task rows)."
    )
    if args.dry_run:
        dry_run(args, galar_root, studies)
        return

    ensure_space(output_root, args.minimum_free_gb)
    state = load_or_create_state(args, output_root, metadata_fingerprints)
    if state.get("already_complete"):
        return
    completed = state["completed_studies"]
    assert isinstance(completed, dict)
    completed_frames = sum(int(item["frames"]) for item in completed.values())
    subsampling = SUBSAMPLING[args.jpeg_subsampling]

    with (
        ThreadPoolExecutor(max_workers=args.workers) as pool,
        tqdm(total=total_frames, initial=completed_frames, unit="frame", desc="SSD cache") as progress,
    ):
        for study_id in sorted(studies, key=int):
            if study_id in completed:
                item = completed[study_id]
                shard = output_root / str(item["shard"])
                index = output_root / str(item["index"])
                if not shard.is_file() or not index.is_file():
                    raise FileNotFoundError(
                        f"Completed study {study_id} is missing cache files; "
                        "do not resume into this directory"
                    )
                if (
                    shard.stat().st_size != int(item["shard_bytes"])
                    or index.stat().st_size != int(item["index_bytes"])
                ):
                    raise ValueError(
                        f"Completed cache files for study {study_id} changed; "
                        "do not resume into this directory"
                    )
                continue

            rows = sorted(studies[study_id].items())
            max_frame = rows[-1][0]
            shard_relative = Path("shards") / f"study_{study_id}.jpgpack"
            index_relative = Path("indexes") / f"study_{study_id}.idx"
            shard_final = output_root / shard_relative
            index_final = output_root / index_relative
            shard_part = shard_final.with_suffix(shard_final.suffix + ".part")
            index_part = index_final.with_suffix(index_final.suffix + ".part")
            # Incomplete per-study files are never referenced and are rebuilt on resume.
            shard_part.unlink(missing_ok=True)
            index_part.unlink(missing_ok=True)

            expected_index_bytes = (max_frame + 1) * INDEX_RECORD.size
            ensure_space(output_root, args.minimum_free_gb, expected_index_bytes)
            with shard_part.open("w+b") as shard_handle, index_part.open("w+b") as index_handle:
                index_handle.truncate(expected_index_bytes)
                for batch in chunks(rows, args.conversion_batch_size):
                    source_paths = [galar_root / Path(relative) for _, relative in batch]
                    encoded_batch = list(
                        pool.map(
                            lambda path: convert_image(
                                path, args.image_size, args.jpeg_quality, subsampling
                            ),
                            source_paths,
                        )
                    )
                    ensure_space(
                        output_root,
                        args.minimum_free_gb,
                        sum(len(encoded) for encoded in encoded_batch),
                    )
                    for (frame_number, _), encoded in zip(
                        batch, encoded_batch, strict=True
                    ):
                        if len(encoded) > 0xFFFFFFFF:
                            raise ValueError("One encoded frame exceeds the uint32 index limit")
                        offset = shard_handle.tell()
                        shard_handle.write(encoded)
                        index_handle.seek(frame_number * INDEX_RECORD.size)
                        index_handle.write(INDEX_RECORD.pack(offset, len(encoded)))
                        progress.update()
                shard_handle.flush()
                index_handle.flush()
                os.fsync(shard_handle.fileno())
                os.fsync(index_handle.fileno())
            shard_part.replace(shard_final)
            index_part.replace(index_final)
            completed[study_id] = {
                "frames": len(rows),
                "max_frame_number": max_frame,
                "shard": shard_relative.as_posix(),
                "shard_bytes": shard_final.stat().st_size,
                "index": index_relative.as_posix(),
                "index_bytes": index_final.stat().st_size,
            }
            write_json_atomic(output_root / "build_state.json", state)

    manifest = {
        "format": CACHE_FORMAT,
        "status": "complete",
        "settings": state["settings"],
        "source_galar_root": str(galar_root),
        "source_metadata_dir": str(metadata_dir),
        "metadata_files": metadata_fingerprints,
        "unique_frames": total_frames,
        "studies": completed,
        "total_jpeg_bytes": sum(int(item["shard_bytes"]) for item in completed.values()),
        "total_index_bytes": sum(int(item["index_bytes"]) for item in completed.values()),
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    (output_root / "build_state.json").unlink()
    total_bytes = int(manifest["total_jpeg_bytes"]) + int(manifest["total_index_bytes"])
    print(
        f"Cache complete: {total_frames:,} frames, {total_bytes / 1024**3:.2f} GiB, "
        f"free space remaining={free_gib(output_root):.2f} GiB"
    )
    print(f"Manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
