"""Canonical label groups from the Galar annotation CSV schema."""

from __future__ import annotations


GI_SECTION_LABELS = (
    "mouth",
    "esophagus",
    "stomach",
    "small intestine",
    "colon",
)

LANDMARK_LABELS = (
    "z-line",
    "pylorus",
    "ampulla of vater",
    "ileocecal valve",
)

ANATOMY_LABELS = LANDMARK_LABELS + GI_SECTION_LABELS

# These columns describe findings rather than location or image quality. The
# preparation script keeps only columns with positive annotations.
PATHOLOGY_CANDIDATE_LABELS = (
    "ulcer",
    "polyp",
    "active bleeding",
    "blood",
    "erythema",
    "erosion",
    "angiectasia",
    "IBD",
    "foreign body",
    "esophagitis",
    "varices",
    "hematin",
    "celiac",
    "cancer",
    "lymphangioectasis",
)

TECHNICAL_LABELS = (
    "bubbles",
    "dirt",
    "no view",
    "reduced view",
    "good view",
)

STRUCTURAL_COLUMNS = ("index", "section", "frame")


def positive_labels(row: dict[str, str], labels: tuple[str, ...]) -> list[str]:
    """Return positive binary labels while validating Galar's 0/1 encoding."""

    active: list[str] = []
    for label in labels:
        value = str(row.get(label, "")).strip()
        if value not in {"0", "1"}:
            raise ValueError(f"Unexpected value {value!r} in Galar column {label!r}")
        if value == "1":
            active.append(label)
    return active
