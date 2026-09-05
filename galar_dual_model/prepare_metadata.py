"""Prepare leak-free multi-label metadata for the two Galar tasks."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from labels import (
    ANATOMY_LABELS,
    LANDMARK_LABELS,
    PATHOLOGY_CANDIDATE_LABELS,
    STRUCTURAL_COLUMNS,
    TECHNICAL_LABELS,
    positive_labels,
)


METADATA_FIELDS = ("image_path", "study_id", "video_id", "frame_number", "labels")
LANDMARK_LABEL_SET = frozenset(LANDMARK_LABELS)


@dataclass
class StudyStats:
    study_id: str
    label_path: Path
    rows: int = 0
    anatomy_counts: Counter[str] = field(default_factory=Counter)
    pathology_counts: Counter[str] = field(default_factory=Counter)
    technical_counts: Counter[str] = field(default_factory=Counter)
    anatomy_multilabel_frames: int = 0
    pathology_multilabel_frames: int = 0
    anatomy_unlabelled_frames: int = 0
    pathology_negative_frames: int = 0
    anatomy_signature_counts: Counter[tuple[str, ...]] = field(default_factory=Counter)
    pathology_signature_counts: Counter[tuple[str, ...]] = field(default_factory=Counter)

    def count(self, label: str) -> int:
        return self.anatomy_counts[label] + self.pathology_counts[label]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create multi-label Anatomy and Pathology metadata from Galar CSV files "
            "using a reproducible source-study split."
        )
    )
    parser.add_argument(
        "--galar-root",
        type=Path,
        required=True,
        help="Galar root containing Labels/, metadata.csv, and <study_id>/ PNG folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "metadata",
        help="Generated metadata directory (default: galar_dual_model/metadata)",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-search-trials",
        type=int,
        default=20_000,
        help="Deterministic candidate study splits evaluated for class balance",
    )
    parser.add_argument(
        "--missing-study-policy",
        choices=("error", "skip"),
        default="error",
        help="How to handle annotation CSVs without a matching PNG study directory",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Check every referenced PNG (slow on the complete dataset)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated metadata set",
    )
    parser.add_argument(
        "--anatomy-train-cap-per-study-labelset",
        type=int,
        default=1_000,
        help=(
            "Evenly spaced train frames retained per study and section-only labelset; "
            "landmark frames are always retained (0 disables the cap)"
        ),
    )
    parser.add_argument(
        "--pathology-train-cap-per-study-labelset",
        type=int,
        default=1_000,
        help=(
            "Evenly spaced train frames retained per study and common positive "
            "pathology labelset (0 disables the cap)"
        ),
    )
    parser.add_argument(
        "--pathology-negative-train-cap-per-study",
        type=int,
        default=1_000,
        help="Evenly spaced all-zero pathology train frames retained per study (0 disables)",
    )
    parser.add_argument(
        "--rare-pathology-max-frames",
        type=int,
        default=5_000,
        help="Preserve every train frame of classes with at most this many positives",
    )
    parser.add_argument(
        "--rare-pathology-max-studies",
        type=int,
        default=3,
        help="Preserve every train frame of classes present in at most this many studies",
    )
    return parser.parse_args()


def natural_study_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def discover_label_files(
    galar_root: Path, missing_study_policy: str
) -> tuple[list[Path], list[str], int]:
    labels_dir = galar_root / "Labels"
    metadata_path = galar_root / "metadata.csv"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Galar Labels directory not found: {labels_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Galar metadata.csv not found: {metadata_path}")

    all_files = sorted(labels_dir.glob("*.csv"), key=natural_study_key)
    if not all_files:
        raise FileNotFoundError(f"No Galar annotation CSV files found in {labels_dir}")

    selected: list[Path] = []
    missing: list[str] = []
    for path in all_files:
        if (galar_root / path.stem).is_dir():
            selected.append(path)
        else:
            missing.append(path.stem)
    if missing and missing_study_policy == "error":
        raise FileNotFoundError(
            "Galar PNG directories are missing for studies "
            + ", ".join(missing)
            + "; download them or use --missing-study-policy skip"
        )
    if len(selected) < 2:
        raise ValueError("At least two available Galar studies are required")
    return selected, missing, len(all_files)


def validate_header(path: Path, fieldnames: Iterable[str] | None) -> None:
    fields = set(fieldnames or ())
    required = set(ANATOMY_LABELS) | set(PATHOLOGY_CANDIDATE_LABELS) | set(
        STRUCTURAL_COLUMNS
    )
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"{path}: missing required Galar columns: {missing}")


def scan_study(path: Path) -> StudyStats:
    stats = StudyStats(study_id=path.stem, label_path=path)
    previous_frame = -1
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_header(path, reader.fieldnames)
        available_technical = tuple(
            label for label in TECHNICAL_LABELS if label in (reader.fieldnames or ())
        )
        for row_number, row in enumerate(reader, start=2):
            try:
                frame_number = int(row["frame"])
                anatomy = positive_labels(row, ANATOMY_LABELS)
                pathology = positive_labels(row, PATHOLOGY_CANDIDATE_LABELS)
                technical = positive_labels(row, available_technical)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{row_number}: {error}") from error
            if frame_number <= previous_frame:
                raise ValueError(
                    f"{path}:{row_number}: frame numbers must be strictly increasing"
                )
            previous_frame = frame_number

            stats.rows += 1
            stats.anatomy_counts.update(anatomy)
            stats.pathology_counts.update(pathology)
            stats.technical_counts.update(technical)
            if anatomy:
                stats.anatomy_signature_counts[tuple(anatomy)] += 1
            stats.pathology_signature_counts[tuple(pathology)] += 1
            stats.anatomy_multilabel_frames += len(anatomy) > 1
            stats.pathology_multilabel_frames += len(pathology) > 1
            stats.anatomy_unlabelled_frames += not anatomy
            stats.pathology_negative_frames += not pathology
    if stats.rows == 0:
        raise ValueError(f"Empty Galar annotation file: {path}")
    return stats


def choose_validation_studies(
    studies: list[StudyStats],
    classes: tuple[str, ...],
    validation_fraction: float,
    seed: int,
    trials: int,
) -> set[str]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if trials < 1:
        raise ValueError("--split-search-trials must be positive")

    study_ids = [study.study_id for study in studies]
    validation_size = max(1, min(len(studies) - 1, round(len(studies) * validation_fraction)))
    support = {
        label: {study.study_id for study in studies if study.count(label) > 0}
        for label in classes
    }
    required_validation = {label for label, ids in support.items() if len(ids) >= 2}
    sole_support_studies = {
        next(iter(ids)) for ids in support.values() if len(ids) == 1
    }
    eligible = [study_id for study_id in study_ids if study_id not in sole_support_studies]
    if len(eligible) < validation_size:
        raise RuntimeError(
            "Not enough studies can be assigned to validation while retaining every class in train"
        )

    total_rows = sum(study.rows for study in studies)
    total_counts = {
        label: sum(study.count(label) for study in studies) for label in classes
    }
    stats_by_id = {study.study_id: study for study in studies}
    rng = random.Random(seed)
    best: tuple[float, tuple[str, ...], set[str]] | None = None

    for _ in range(trials):
        candidate: set[str] = set()
        uncovered = set(required_validation)
        while uncovered and len(candidate) < validation_size:
            gains: list[tuple[float, str]] = []
            for study_id in eligible:
                if study_id in candidate:
                    continue
                covered = [
                    label
                    for label in classes
                    if label in uncovered and study_id in support[label]
                ]
                gain = sum(1.0 / len(support[label]) for label in covered)
                if gain:
                    gains.append((gain * (0.95 + 0.10 * rng.random()), study_id))
            if not gains:
                break
            chosen = max(gains)[1]
            candidate.add(chosen)
            uncovered -= {
                label for label in classes if chosen in support[label]
            }

        remaining = [study_id for study_id in eligible if study_id not in candidate]
        if len(candidate) > validation_size or len(remaining) < validation_size - len(candidate):
            continue
        candidate.update(rng.sample(remaining, validation_size - len(candidate)))

        missing_validation = sum(
            not (ids & candidate) for ids in support.values() if len(ids) >= 2
        )
        missing_train = sum(ids <= candidate for ids in support.values())
        if missing_validation or missing_train:
            continue

        validation_rows = sum(stats_by_id[study_id].rows for study_id in candidate)
        balance_error = abs(validation_rows / total_rows - validation_fraction)
        for label in classes:
            label_total = total_counts[label]
            label_validation = sum(
                stats_by_id[study_id].count(label) for study_id in candidate
            )
            frame_ratio = label_validation / label_total
            study_ratio = len(support[label] & candidate) / len(support[label])
            balance_error += abs(frame_ratio - validation_fraction)
            balance_error += 0.25 * abs(study_ratio - validation_fraction)

        ordered_candidate = tuple(sorted(candidate, key=int))
        result = (balance_error, ordered_candidate, candidate)
        if best is None or result[:2] < best[:2]:
            best = result

    if best is None:
        raise RuntimeError(
            "Could not find a study-level split with train/validation class support; "
            "increase --split-search-trials or change --validation-fraction"
        )
    return best[2]


def evenly_spaced_positions(total: int, cap: int) -> set[int] | None:
    """Return zero-based positions spanning a temporal group, or None to keep all."""

    if cap == 0 or total <= cap:
        return None
    if cap == 1:
        return {total // 2}
    return {
        round(index * (total - 1) / (cap - 1))
        for index in range(cap)
    }


def train_pathology_support(
    studies: list[StudyStats],
    validation_studies: set[str],
    classes: tuple[str, ...],
) -> tuple[Counter[str], Counter[str]]:
    frame_counts: Counter[str] = Counter()
    study_counts: Counter[str] = Counter()
    for study in studies:
        if study.study_id in validation_studies:
            continue
        frame_counts.update(study.pathology_counts)
        study_counts.update(
            label for label in classes if study.pathology_counts[label] > 0
        )
    return frame_counts, study_counts


def task_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "anatomy_train": output_dir / "anatomy_train.csv",
        "anatomy_validation": output_dir / "anatomy_validation.csv",
        "pathology_train": output_dir / "pathology_train.csv",
        "pathology_validation": output_dir / "pathology_validation.csv",
        "anatomy_config": output_dir / "anatomy_config.json",
        "pathology_config": output_dir / "pathology_config.json",
        "split": output_dir / "split.json",
        "summary": output_dir / "summary.json",
    }


def ensure_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Generated metadata already exists; use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )


def empty_task_stats(classes: tuple[str, ...]) -> dict[str, object]:
    return {
        "frames": 0,
        "positive_frames": 0,
        "negative_frames": 0,
        "multilabel_frames": 0,
        "class_frame_counts": Counter({label: 0 for label in classes}),
        "class_studies": {label: set() for label in classes},
        "studies": set(),
    }


def update_task_stats(
    stats: dict[str, object], study_id: str, labels: list[str]
) -> None:
    stats["frames"] = int(stats["frames"]) + 1
    stats["positive_frames"] = int(stats["positive_frames"]) + bool(labels)
    stats["negative_frames"] = int(stats["negative_frames"]) + (not labels)
    stats["multilabel_frames"] = int(stats["multilabel_frames"]) + (len(labels) > 1)
    frame_counts = stats["class_frame_counts"]
    studies = stats["studies"]
    class_studies = stats["class_studies"]
    assert isinstance(frame_counts, Counter)
    assert isinstance(studies, set)
    assert isinstance(class_studies, dict)
    frame_counts.update(labels)
    studies.add(study_id)
    for label in labels:
        class_studies[label].add(study_id)


def serialise_task_stats(
    stats: dict[str, object], classes: tuple[str, ...]
) -> dict[str, object]:
    frame_counts = stats["class_frame_counts"]
    studies = stats["studies"]
    class_studies = stats["class_studies"]
    assert isinstance(frame_counts, Counter)
    assert isinstance(studies, set)
    assert isinstance(class_studies, dict)
    return {
        "frames": int(stats["frames"]),
        "studies": len(studies),
        "positive_frames": int(stats["positive_frames"]),
        "negative_frames": int(stats["negative_frames"]),
        "multilabel_frames": int(stats["multilabel_frames"]),
        "class_frame_counts": {label: frame_counts[label] for label in classes},
        "class_study_counts": {label: len(class_studies[label]) for label in classes},
    }


def write_metadata(
    galar_root: Path,
    studies: list[StudyStats],
    validation_studies: set[str],
    anatomy_classes: tuple[str, ...],
    pathology_classes: tuple[str, ...],
    paths: dict[str, Path],
    verify_images: bool,
    anatomy_train_cap: int,
    pathology_train_cap: int,
    pathology_negative_train_cap: int,
    rare_pathology_classes: set[str],
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, dict[str, object]]]:
    csv_keys = (
        "anatomy_train",
        "anatomy_validation",
        "pathology_train",
        "pathology_validation",
    )
    temporary = {key: paths[key].with_suffix(".csv.part") for key in csv_keys}
    handles = {
        key: temporary[key].open("w", encoding="utf-8", newline="") for key in csv_keys
    }
    writers = {
        key: csv.DictWriter(handles[key], fieldnames=METADATA_FIELDS) for key in csv_keys
    }
    for writer in writers.values():
        writer.writeheader()

    raw_stats = {
        "anatomy": {
            "train": empty_task_stats(anatomy_classes),
            "validation": empty_task_stats(anatomy_classes),
        },
        "pathology": {
            "train": empty_task_stats(pathology_classes),
            "validation": empty_task_stats(pathology_classes),
        },
    }
    try:
        for study in studies:
            split = "validation" if study.study_id in validation_studies else "train"
            anatomy_positions = {
                signature: evenly_spaced_positions(count, anatomy_train_cap)
                for signature, count in study.anatomy_signature_counts.items()
                if not LANDMARK_LABEL_SET.intersection(signature)
            }
            pathology_positions = {
                signature: evenly_spaced_positions(
                    count,
                    pathology_negative_train_cap
                    if not signature
                    else pathology_train_cap,
                )
                for signature, count in study.pathology_signature_counts.items()
                if not set(signature).intersection(rare_pathology_classes)
            }
            anatomy_seen: Counter[tuple[str, ...]] = Counter()
            pathology_seen: Counter[tuple[str, ...]] = Counter()
            with study.label_path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                validate_header(study.label_path, reader.fieldnames)
                for row_number, row in enumerate(reader, start=2):
                    try:
                        frame_number = int(row["frame"])
                        anatomy = positive_labels(row, anatomy_classes)
                        pathology = positive_labels(row, pathology_classes)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"{study.label_path}:{row_number}: {error}"
                        ) from error

                    anatomy_signature = tuple(anatomy)
                    pathology_signature = tuple(pathology)
                    anatomy_position = anatomy_seen[anatomy_signature]
                    pathology_position = pathology_seen[pathology_signature]
                    anatomy_seen[anatomy_signature] += 1
                    pathology_seen[pathology_signature] += 1
                    keep_anatomy = bool(anatomy)
                    keep_pathology = True
                    if split == "train":
                        anatomy_selected = anatomy_positions.get(anatomy_signature)
                        if (
                            anatomy
                            and not LANDMARK_LABEL_SET.intersection(anatomy)
                            and anatomy_selected is not None
                        ):
                            keep_anatomy = anatomy_position in anatomy_selected
                        pathology_selected = pathology_positions.get(pathology_signature)
                        if (
                            not set(pathology).intersection(rare_pathology_classes)
                            and pathology_selected is not None
                        ):
                            keep_pathology = pathology_position in pathology_selected

                    relative_image = Path(study.study_id) / f"frame_{frame_number:06d}.PNG"
                    if (
                        verify_images
                        and (keep_anatomy or keep_pathology)
                        and not (galar_root / relative_image).is_file()
                    ):
                        raise FileNotFoundError(
                            f"Annotated Galar image is missing: {galar_root / relative_image}"
                        )
                    common = {
                        "image_path": relative_image.as_posix(),
                        "study_id": study.study_id,
                        "video_id": f"galar_{study.study_id}",
                        "frame_number": frame_number,
                    }
                    if keep_anatomy:
                        writers[f"anatomy_{split}"].writerow(
                            {**common, "labels": "|".join(anatomy)}
                        )
                        update_task_stats(raw_stats["anatomy"][split], study.study_id, anatomy)
                    if keep_pathology:
                        writers[f"pathology_{split}"].writerow(
                            {**common, "labels": "|".join(pathology)}
                        )
                        update_task_stats(
                            raw_stats["pathology"][split], study.study_id, pathology
                        )
    except Exception:
        for handle in handles.values():
            handle.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    else:
        for handle in handles.values():
            handle.close()
        for key in csv_keys:
            temporary[key].replace(paths[key])

    stats = {
        task: {
            split: serialise_task_stats(raw_stats[task][split], classes)
            for split in ("train", "validation")
        }
        for task, classes in (
            ("anatomy", anatomy_classes),
            ("pathology", pathology_classes),
        )
    }
    raw_anatomy_train_frames = sum(
        study.rows - study.anatomy_unlabelled_frames
        for study in studies
        if study.study_id not in validation_studies
    )
    raw_pathology_train_frames = sum(
        study.rows for study in studies if study.study_id not in validation_studies
    )
    raw_anatomy_counts = Counter()
    raw_pathology_counts = Counter()
    for study in studies:
        if study.study_id not in validation_studies:
            raw_anatomy_counts.update(study.anatomy_counts)
            raw_pathology_counts.update(study.pathology_counts)
    sampling = {
        "anatomy": {
            "policy": "cap section-only frames per (study, exact labelset); preserve all landmarks",
            "cap_per_study_labelset": anatomy_train_cap,
            "raw_train_frames": raw_anatomy_train_frames,
            "retained_train_frames": stats["anatomy"]["train"]["frames"],
            "dropped_train_frames": (
                raw_anatomy_train_frames - stats["anatomy"]["train"]["frames"]
            ),
            "raw_class_frame_counts": {
                label: raw_anatomy_counts[label] for label in anatomy_classes
            },
        },
        "pathology": {
            "policy": (
                "cap all-zero frames per study and common positive frames per "
                "(study, exact labelset); preserve every frame containing a rare class"
            ),
            "positive_cap_per_study_labelset": pathology_train_cap,
            "negative_cap_per_study": pathology_negative_train_cap,
            "rare_classes_preserved": sorted(rare_pathology_classes),
            "raw_train_frames": raw_pathology_train_frames,
            "retained_train_frames": stats["pathology"]["train"]["frames"],
            "dropped_train_frames": (
                raw_pathology_train_frames - stats["pathology"]["train"]["frames"]
            ),
            "raw_class_frame_counts": {
                label: raw_pathology_counts[label] for label in pathology_classes
            },
        },
    }
    return stats, sampling


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def print_task_summary(
    task: str, classes: tuple[str, ...], stats: dict[str, dict[str, object]]
) -> None:
    print(f"\n{task.upper()} ({len(classes)} classes, multi-label)")
    print("class\ttrain frames/studies\tvalidation frames/studies")
    for label in classes:
        print(
            f"{label}\t"
            f"{stats['train']['class_frame_counts'][label]}/"
            f"{stats['train']['class_study_counts'][label]}\t"
            f"{stats['validation']['class_frame_counts'][label]}/"
            f"{stats['validation']['class_study_counts'][label]}"
        )
    for split in ("train", "validation"):
        part = stats[split]
        print(
            f"{split}: frames={part['frames']}, studies={part['studies']}, "
            f"positive={part['positive_frames']}, negative={part['negative_frames']}, "
            f"multi_label={part['multilabel_frames']}"
        )
    unsupported = [
        label
        for label in classes
        if stats["validation"]["class_frame_counts"][label] == 0
    ]
    print("Classes without validation support:", unsupported or "none")


def main() -> None:
    args = parse_args()
    cap_values = {
        "--anatomy-train-cap-per-study-labelset": args.anatomy_train_cap_per_study_labelset,
        "--pathology-train-cap-per-study-labelset": args.pathology_train_cap_per_study_labelset,
        "--pathology-negative-train-cap-per-study": (
            args.pathology_negative_train_cap_per_study
        ),
        "--rare-pathology-max-frames": args.rare_pathology_max_frames,
        "--rare-pathology-max-studies": args.rare_pathology_max_studies,
    }
    invalid_caps = [name for name, value in cap_values.items() if value < 0]
    if invalid_caps:
        raise ValueError("Sampling limits cannot be negative: " + ", ".join(invalid_caps))
    galar_root = args.galar_root.resolve()
    label_files, missing_studies, total_annotation_files = discover_label_files(
        galar_root, args.missing_study_policy
    )
    print(
        f"Scanning {len(label_files)}/{total_annotation_files} Galar studies "
        f"with locally available PNG directories..."
    )
    studies = [scan_study(path) for path in label_files]

    anatomy_counts = Counter()
    pathology_counts = Counter()
    technical_counts = Counter()
    for study in studies:
        anatomy_counts.update(study.anatomy_counts)
        pathology_counts.update(study.pathology_counts)
        technical_counts.update(study.technical_counts)

    anatomy_classes = tuple(label for label in ANATOMY_LABELS if anatomy_counts[label] > 0)
    pathology_classes = tuple(
        label for label in PATHOLOGY_CANDIDATE_LABELS if pathology_counts[label] > 0
    )
    zero_support_pathology = tuple(
        label for label in PATHOLOGY_CANDIDATE_LABELS if pathology_counts[label] == 0
    )
    if not anatomy_classes or not pathology_classes:
        raise ValueError("Selected Galar studies do not contain both Anatomy and Pathology labels")

    validation_studies = choose_validation_studies(
        studies,
        anatomy_classes + pathology_classes,
        args.validation_fraction,
        args.seed,
        args.split_search_trials,
    )
    pathology_train_frames, pathology_train_studies = train_pathology_support(
        studies, validation_studies, pathology_classes
    )
    rare_pathology_classes = {
        label
        for label in pathology_classes
        if (
            args.rare_pathology_max_frames > 0
            and pathology_train_frames[label] <= args.rare_pathology_max_frames
        )
        or (
            args.rare_pathology_max_studies > 0
            and pathology_train_studies[label] <= args.rare_pathology_max_studies
        )
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = task_paths(output_dir)
    ensure_outputs(paths, args.overwrite)
    stats, sampling = write_metadata(
        galar_root,
        studies,
        validation_studies,
        anatomy_classes,
        pathology_classes,
        paths,
        args.verify_images,
        args.anatomy_train_cap_per_study_labelset,
        args.pathology_train_cap_per_study_labelset,
        args.pathology_negative_train_cap_per_study,
        rare_pathology_classes,
    )

    train_studies = sorted(
        (study.study_id for study in studies if study.study_id not in validation_studies),
        key=int,
    )
    ordered_validation = sorted(validation_studies, key=int)
    split_payload = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "split_unit": "source_study_id",
        "train_studies": train_studies,
        "validation_studies": ordered_validation,
    }
    write_json(paths["split"], split_payload)

    for task, classes in (("anatomy", anatomy_classes), ("pathology", pathology_classes)):
        task_payload = {
            "task": task,
            "problem_type": "multi_label",
            "classes": list(classes),
            "threshold": 0.5,
            "train_file": paths[f"{task}_train"].name,
            "validation_file": paths[f"{task}_validation"].name,
            "split_file": paths["split"].name,
            "training_sampling": sampling[task],
            "train": stats[task]["train"],
            "validation": stats[task]["validation"],
        }
        write_json(paths[f"{task}_config"], task_payload)

    summary = {
        "source": "Galar",
        "galar_root": str(galar_root),
        "annotation_studies": total_annotation_files,
        "included_studies": len(studies),
        "omitted_missing_png_studies": missing_studies,
        "split": split_payload,
        "tasks": {
            "anatomy": {
                "problem_type": "multi_label",
                "classes": list(anatomy_classes),
                "reason": "GI-section labels overlap valid anatomical landmark labels.",
                "training_sampling": sampling["anatomy"],
                **stats["anatomy"],
            },
            "pathology": {
                "problem_type": "multi_label",
                "classes": list(pathology_classes),
                "zero_support_excluded_classes": list(zero_support_pathology),
                "reason": "Multiple pathological findings co-occur on individual frames.",
                "training_sampling": sampling["pathology"],
                **stats["pathology"],
            },
        },
        "excluded": {
            "technical_quality_labels": {
                label: technical_counts[label] for label in TECHNICAL_LABELS
            },
            "structural_or_redundant_columns": list(STRUCTURAL_COLUMNS),
        },
        "image_verification": args.verify_images,
    }
    write_json(paths["summary"], summary)

    print(f"Metadata written to: {output_dir}")
    if missing_studies:
        print("Omitted studies without local PNG folders:", ", ".join(missing_studies))
    for task in ("anatomy", "pathology"):
        task_sampling = sampling[task]
        print(
            f"{task} train temporal sampling: "
            f"{task_sampling['retained_train_frames']}/"
            f"{task_sampling['raw_train_frames']} frames retained "
            f"({task_sampling['dropped_train_frames']} dropped)"
        )
    print(
        "Pathology classes preserved without caps:",
        ", ".join(sorted(rare_pathology_classes)) or "none",
    )
    print_task_summary("anatomy", anatomy_classes, stats["anatomy"])
    print_task_summary("pathology", pathology_classes, stats["pathology"])


if __name__ == "__main__":
    main()
