"""Select Galar studies needed for the three under-supported Kvasir classes.

The script scans ``another db/Labels/*.csv`` without loading the full 246 MB
annotation set into memory.  It writes one manifest row per Galar video that
contains at least one usable target frame.

A target frame is usable for the current single-label classifier when exactly
one of the three requested targets is positive and no other pathological Galar
label is positive.  Anatomical landmarks, GI-section labels, and the free-text
``section`` field are intentionally ignored when checking conflicts.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


TARGET_COLUMNS = {
    "ampulla of vater": "ampulla_of_vater",
    "hematin": "blood_hematin",
    "polyp": "polyp",
}

# These are alternative pathological findings, not anatomy/technical labels.
# A positive value beside a requested target makes the frame ambiguous for the
# current single-label softmax task.  In particular, generic blood/bleeding is
# never silently converted to blood_hematin.
OTHER_PATHOLOGY_COLUMNS = (
    "ulcer",
    "active bleeding",
    "blood",
    "erythema",
    "erosion",
    "angiectasia",
    "IBD",
    "foreign body",
    "esophagitis",
    "varices",
    "celiac",
    "cancer",
    "lymphangioectasis",
)


def is_positive(value: str | None) -> bool:
    """Return True for the binary positive representation used by Galar."""

    return str(value or "").strip() == "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manifest of Galar videos needed for rare Kvasir classes."
    )
    parser.add_argument(
        "--galar-root",
        type=Path,
        default=Path("another db"),
        help="Directory containing metadata.csv and Labels/ (default: another db)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("another db/galar_target_videos.csv"),
        help="Output CSV manifest (default: another db/galar_target_videos.csv)",
    )
    return parser.parse_args()


def read_study_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"File Name", "Capsule System", "Gender ", "Age"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"Missing Galar metadata columns: {missing}")
        return {row["File Name"].strip(): row for row in reader}


def analyze_label_file(path: Path) -> dict[str, object]:
    raw_counts: Counter[str] = Counter()
    usable_counts: Counter[str] = Counter()
    ambiguous_target_counts: Counter[str] = Counter()
    min_frames: dict[str, int] = {}
    max_frames: dict[str, int] = {}
    row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(TARGET_COLUMNS) | set(OTHER_PATHOLOGY_COLUMNS) | {"frame"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"{path}: missing columns: {missing}")

        for row in reader:
            row_count += 1
            active_targets = [
                target for source, target in TARGET_COLUMNS.items() if is_positive(row[source])
            ]
            if not active_targets:
                continue

            frame = int(row["frame"])
            for target in active_targets:
                raw_counts[target] += 1
                min_frames[target] = min(frame, min_frames.get(target, frame))
                max_frames[target] = max(frame, max_frames.get(target, frame))

            has_other_pathology = any(
                is_positive(row[column]) for column in OTHER_PATHOLOGY_COLUMNS
            )
            if len(active_targets) == 1 and not has_other_pathology:
                usable_counts[active_targets[0]] += 1
            else:
                ambiguous_target_counts.update(active_targets)

    return {
        "annotation_rows": row_count,
        "raw_counts": raw_counts,
        "usable_counts": usable_counts,
        "ambiguous_counts": ambiguous_target_counts,
        "min_frames": min_frames,
        "max_frames": max_frames,
    }


def build_manifest(galar_root: Path) -> list[dict[str, object]]:
    labels_dir = galar_root / "Labels"
    metadata_path = galar_root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Galar metadata not found: {metadata_path}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Galar label directory not found: {labels_dir}")

    studies = read_study_metadata(metadata_path)
    label_files = sorted(labels_dir.glob("*.csv"), key=lambda item: int(item.stem))
    if not label_files:
        raise FileNotFoundError(f"No label CSV files found in {labels_dir}")

    manifest: list[dict[str, object]] = []
    seen_studies: set[str] = set()
    for label_path in label_files:
        study_id = label_path.stem
        if study_id not in studies:
            raise ValueError(f"{label_path}: study {study_id!r} is absent from metadata.csv")
        seen_studies.add(study_id)
        analysis = analyze_label_file(label_path)
        usable = analysis["usable_counts"]
        assert isinstance(usable, Counter)
        if not any(usable[target] for target in TARGET_COLUMNS.values()):
            continue

        raw = analysis["raw_counts"]
        ambiguous = analysis["ambiguous_counts"]
        min_frames = analysis["min_frames"]
        max_frames = analysis["max_frames"]
        assert isinstance(raw, Counter)
        assert isinstance(ambiguous, Counter)
        assert isinstance(min_frames, dict)
        assert isinstance(max_frames, dict)

        source = studies[study_id]
        usable_targets = [target for target in TARGET_COLUMNS.values() if usable[target]]
        row: dict[str, object] = {
            "source_dataset": "Galar",
            "study_id": study_id,
            "video_id": f"galar_{study_id}",
            "capsule_system": source["Capsule System"].strip(),
            "gender": source["Gender "].strip(),
            "age": source["Age"].strip(),
            "label_file": label_path.relative_to(galar_root).as_posix(),
            "target_classes": "|".join(usable_targets),
            "annotation_rows": analysis["annotation_rows"],
        }
        for target in TARGET_COLUMNS.values():
            row[f"{target}_raw_frames"] = raw[target]
            row[f"{target}_usable_frames"] = usable[target]
            row[f"{target}_ambiguous_frames"] = ambiguous[target]
            row[f"{target}_first_frame"] = min_frames.get(target, "")
            row[f"{target}_last_frame"] = max_frames.get(target, "")
        manifest.append(row)

    missing_label_files = sorted(set(studies) - seen_studies, key=int)
    if missing_label_files:
        raise ValueError(
            "metadata.csv contains studies without label files: "
            + ", ".join(missing_label_files)
        )
    return manifest


def write_manifest(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        raise ValueError("No Galar videos contain usable target frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]], output: Path) -> None:
    print(f"Selected {len(rows)} Galar videos -> {output}")
    print("video_id\ttarget_classes\tampulla usable/raw\thematin usable/raw\tpolyp usable/raw")
    for row in rows:
        print(
            f"{row['video_id']}\t{row['target_classes']}\t"
            f"{row['ampulla_of_vater_usable_frames']}/{row['ampulla_of_vater_raw_frames']}\t"
            f"{row['blood_hematin_usable_frames']}/{row['blood_hematin_raw_frames']}\t"
            f"{row['polyp_usable_frames']}/{row['polyp_raw_frames']}"
        )
    totals = {
        target: sum(int(row[f"{target}_usable_frames"]) for row in rows)
        for target in TARGET_COLUMNS.values()
    }
    print("Usable target frames:", ", ".join(f"{k}={v}" for k, v in totals.items()))


def main() -> None:
    args = parse_args()
    rows = build_manifest(args.galar_root)
    write_manifest(rows, args.output)
    print_summary(rows, args.output)


if __name__ == "__main__":
    main()
