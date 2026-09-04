"""Regularized Kvasir-Capsule frame classifier with video-level validation."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import v2
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_label(value: str) -> str:
    return value.strip().lower().replace(" - ", "_").replace(" ", "_").replace("-", "_")


def index_images(root: Path) -> dict[tuple[str, str], Path]:
    """Index by (filename, class folder); one frame may legitimately have multiple labels."""
    paths: dict[tuple[str, str], Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            key = (path.name, normalize_label(path.parent.name))
            if key in paths:
                raise RuntimeError(f"Duplicate image within class folder: {path}")
            paths[key] = path
    return paths


def choose_group_split(metadata: pd.DataFrame, val_size: float, seed: int):
    """Pick a leak-free split while keeping every class represented in training."""
    all_targets = set(metadata.target.unique())
    videos_per_target = metadata.groupby("target").video_id.nunique()
    validation_eligible = set(videos_per_target[videos_per_target > 1].index)
    total_by_target = metadata.target.value_counts()
    best = None
    for offset in range(500):
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + offset)
        train_idx, val_idx = next(splitter.split(metadata, groups=metadata.video_id))
        train_targets = set(metadata.iloc[train_idx].target.unique())
        if train_targets != all_targets:
            continue
        validation = metadata.iloc[val_idx]
        val_coverage = len(set(validation.target.unique()) & validation_eligible)
        val_fraction_by_target = validation.target.value_counts().reindex(total_by_target.index, fill_value=0) / total_by_target
        eligible_fraction_error = 0.0
        if validation_eligible:
            eligible_fraction_error = float(
                (val_fraction_by_target.loc[list(validation_eligible)] - val_size).abs().mean()
            )
        size_error = abs(len(val_idx) / len(metadata) - val_size)
        score = (val_coverage, -eligible_fraction_error, -size_error)
        if best is None or score > best[0]:
            best = (score, train_idx, val_idx)
    if best is None:
        raise RuntimeError("Could not create a video-level split containing every class in train")
    return best[1], best[2]


def create_rare_diagnostic_split(
    train_df: pd.DataFrame,
    diagnostic_classes: list[str],
    fraction: float,
    gap_frames: int,
    min_diagnostic: int,
    min_train: int,
):
    """Hold out a temporal tail from one-video classes for an explicitly in-video diagnostic."""
    working = train_df.copy()
    working["_frame_order"] = pd.to_numeric(working.frame_number, errors="coerce")
    if working._frame_order.isna().any():
        raise ValueError("frame_number must be numeric to create a temporal diagnostic split")

    reserved = pd.Series(False, index=working.index)
    diagnostic_indices: list[int] = []
    details: dict[str, dict] = {}

    for label in diagnostic_classes:
        candidates = working[working.label == label].sort_values(["_frame_order", "filename"])
        if candidates.video_id.nunique() != 1:
            details[label] = {"status": "unavailable", "reason": "class_is_not_from_exactly_one_video"}
            continue

        desired = max(min_diagnostic, math.ceil(len(candidates) * fraction))
        desired = min(desired, len(candidates) - min_train)
        selected = None
        selected_exclusion = None

        for diagnostic_count in range(desired, min_diagnostic - 1, -1):
            if diagnostic_count <= 0:
                continue
            candidate_diagnostic = candidates.tail(diagnostic_count)
            video_id = candidate_diagnostic.video_id.iloc[0]
            diagnostic_start = int(candidate_diagnostic._frame_order.min())
            diagnostic_end = int(candidate_diagnostic._frame_order.max())
            exclusion = (
                (working.video_id == video_id)
                & (working._frame_order >= diagnostic_start - gap_frames)
                & (working._frame_order <= diagnostic_end + gap_frames)
            )
            remaining_positives = candidates.index.difference(working.index[exclusion])
            if len(remaining_positives) >= min_train:
                selected = candidate_diagnostic
                selected_exclusion = exclusion
                break

        if selected is None or selected_exclusion is None:
            details[label] = {
                "status": "unavailable",
                "reason": "not_enough_temporally_separated_frames",
                "available_positive_frames": int(len(candidates)),
            }
            continue

        reserved |= selected_exclusion
        diagnostic_indices.extend(selected.index.tolist())
        details[label] = {
            "status": "diagnostic_only",
            "video_id": str(selected.video_id.iloc[0]),
            "diagnostic_positive_frames": int(len(selected)),
            "diagnostic_frame_start": int(selected._frame_order.min()),
            "diagnostic_frame_end": int(selected._frame_order.max()),
            "guard_gap_frames": int(gap_frames),
            "excluded_rows_including_diagnostic": int(selected_exclusion.sum()),
        }

    diagnostic_mask = working.index.isin(diagnostic_indices)
    diagnostic_df = working.loc[diagnostic_mask].copy()
    guard_df = working.loc[reserved & ~diagnostic_mask].copy()
    fit_train_df = working.loc[~reserved].copy()

    for label, detail in details.items():
        if detail["status"] == "diagnostic_only":
            detail["remaining_train_positive_frames"] = int((fit_train_df.label == label).sum())

    columns_to_drop = ["_frame_order"]
    return (
        fit_train_df.drop(columns=columns_to_drop),
        diagnostic_df.drop(columns=columns_to_drop),
        guard_df.drop(columns=columns_to_drop),
        details,
    )


def count_by_class(frame: pd.DataFrame, classes: list[str]) -> dict[str, int]:
    counts = frame.label.value_counts()
    return {name: int(counts.get(name, 0)) for name in classes}


def imbalance_ratio(counts: dict[str, int]) -> float | None:
    nonzero = [value for value in counts.values() if value > 0]
    if not nonzero:
        return None
    return float(max(nonzero) / min(nonzero))


def cap_correlated_training_frames(
    frame: pd.DataFrame, max_frames_per_video_class: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep evenly spaced frames so long near-duplicate sequences cannot dominate training."""
    if max_frames_per_video_class == 0:
        return frame.copy(), frame.iloc[0:0].copy()

    ordered = frame.copy()
    ordered["_frame_order"] = pd.to_numeric(ordered.frame_number, errors="coerce")
    keep_indices: list[int] = []
    for _, group in ordered.sort_values(
        ["video_id", "label", "_frame_order", "filename"], na_position="last"
    ).groupby(["video_id", "label"], sort=False):
        if len(group) <= max_frames_per_video_class:
            keep_indices.extend(group.index.tolist())
            continue
        positions = np.linspace(
            0, len(group) - 1, num=max_frames_per_video_class, dtype=int
        )
        keep_indices.extend(group.iloc[positions].index.tolist())

    keep_mask = frame.index.isin(keep_indices)
    return frame.loc[keep_mask].copy(), frame.loc[~keep_mask].copy()


def save_class_distribution(
    train_df: pd.DataFrame, val_df: pd.DataFrame, diagnostic_df: pd.DataFrame, output: Path
) -> None:
    counts = pd.concat(
        [
            train_df.label.value_counts().rename("train"),
            val_df.label.value_counts().rename("validation"),
            diagnostic_df.label.value_counts().rename("diagnostic_only"),
        ],
        axis=1,
    ).fillna(0).sort_index()
    ax = counts.plot(
        kind="bar", figsize=(14, 6), logy=True, color=["#2563eb", "#f97316", "#a855f7"]
    )
    ax.set_title("Class distribution (log scale)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output / "class_distribution.png", dpi=160)
    plt.close()


def save_history_plot(history: list[dict], output: Path) -> None:
    data = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(data.epoch, data.train_loss, marker="o", label="train")
    axes[0].plot(data.epoch, data.val_loss, marker="o", label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(data.epoch, data.train_macro_f1, marker="o", label="train")
    axes[1].plot(data.epoch, data.val_supported_macro_f1, marker="o", label="validation")
    axes[1].set(title="Supported-class macro-F1", xlabel="Epoch", ylabel="F1", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)


def save_confusion_plot(matrix: np.ndarray, names: list[str], output: Path) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set(title="Normalized confusion matrix", xlabel="Predicted", ylabel="True")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    fig.colorbar(image, ax=ax, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=170)
    plt.close(fig)


class CapsuleDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: v2.Compose):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            image = image.convert("RGB")
        return self.transform(image), int(row.target)


def build_loaders(
    train_df,
    val_df,
    diagnostic_df,
    batch_size,
    workers,
    sampler_power,
    sampler_max_multiplier,
    random_erasing_probability,
):
    weights = EfficientNet_B0_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std
    train_tf = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(
            (224, 224), scale=(0.65, 1.0), ratio=(0.85, 1.15), antialias=True
        ),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(20),
        v2.ColorJitter(0.2, 0.2, 0.12, 0.04),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.15),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
        v2.RandomErasing(
            p=random_erasing_probability,
            scale=(0.02, 0.12),
            ratio=(0.5, 2.0),
            value="random",
        ),
    ])
    val_tf = v2.Compose([
        v2.ToImage(), v2.Resize(256), v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std),
    ])
    counts = {int(target): int(count) for target, count in train_df.target.value_counts().items()}
    largest_class = max(counts.values())
    class_weights = {
        target: min((largest_class / count) ** sampler_power, sampler_max_multiplier)
        for target, count in counts.items()
    }
    sample_weights = torch.tensor([class_weights[int(x)] for x in train_df.target], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(CapsuleDataset(train_df, train_tf), sampler=sampler, **common)
    val_loader = DataLoader(CapsuleDataset(val_df, val_tf), shuffle=False, **common)
    diagnostic_loader = None
    if len(diagnostic_df):
        diagnostic_loader = DataLoader(CapsuleDataset(diagnostic_df, val_tf), shuffle=False, **common)

    total_sampling_mass = sum(counts[target] * class_weights[target] for target in counts)
    sampler_details = {
        target: {
            "train_frames": counts[target],
            "per_sample_weight_multiplier": float(class_weights[target]),
            "expected_samples_per_epoch": float(
                len(train_df) * counts[target] * class_weights[target] / total_sampling_mass
            ),
            "expected_repeats_per_frame": float(
                len(train_df) * class_weights[target] / total_sampling_mass
            ),
        }
        for target in counts
    }
    return train_loader, val_loader, diagnostic_loader, sampler_details


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, metric_labels=None):
    training = optimizer is not None
    model.train(training)
    if training and hasattr(model, "features"):
        # Frozen BatchNorm layers must not update running statistics. This matters both while
        # the whole backbone is frozen and when only its final blocks are fine-tuned.
        for block in model.features:
            if not any(parameter.requires_grad for parameter in block.parameters()):
                block.eval()
    total_loss, truth, predicted, confidence = 0.0, [], [], []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, targets in tqdm(loader, leave=False):
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, targets)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * targets.size(0)
            truth.extend(targets.cpu().tolist())
            probabilities = logits.softmax(dim=1)
            batch_confidence, batch_predicted = probabilities.max(dim=1)
            predicted.extend(batch_predicted.cpu().tolist())
            confidence.extend(batch_confidence.cpu().tolist())
    macro_f1 = f1_score(
        truth,
        predicted,
        labels=metric_labels,
        average="macro",
        zero_division=0,
    )
    return total_loss / len(loader.dataset), macro_f1, truth, predicted, confidence


def save_validation_outputs(
    val_df: pd.DataFrame,
    truth: list[int],
    predicted: list[int],
    confidence: list[float],
    class_to_idx: dict[str, int],
    output: Path,
) -> dict:
    names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    labels = list(range(len(names)))
    supported_labels = sorted(set(truth))
    unsupported_labels = sorted(set(labels) - set(supported_labels))
    supported_names = [names[target] for target in supported_labels]
    unsupported_names = [names[target] for target in unsupported_labels]

    supported_report = classification_report(
        truth,
        predicted,
        labels=supported_labels,
        target_names=supported_names,
        zero_division=0,
        output_dict=True,
    )
    unsupported_false_positives = {}
    for target in unsupported_labels:
        false_positives = sum(value == target for value in predicted)
        unsupported_false_positives[names[target]] = {
            "validation_positives": 0,
            "predicted_count": int(false_positives),
            "false_positive_rate": float(false_positives / len(predicted)),
            "sensitivity_status": "not_estimable_no_positive_video",
        }

    report = {
        "evaluation_scope": "leak_free_video_level_validation",
        "model_selection_metric": "supported_macro_f1",
        "supported_classes": supported_names,
        "unsupported_classes": unsupported_names,
        "class_status": {
            name: (
                "evaluated_on_unseen_videos"
                if target in supported_labels
                else "sensitivity_not_estimable_no_positive_video"
            )
            for name, target in class_to_idx.items()
        },
        "supported_classification_report": supported_report,
        "supported_macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=supported_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "supported_balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "overall_accuracy": float(accuracy_score(truth, predicted)),
        "unsupported_class_false_positives": unsupported_false_positives,
        "note": (
            "Unsupported classes have no positive video in validation. Their sensitivity is not "
            "estimable; only false positives on validation negatives are reported."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions = val_df[["filename", "video_id", "frame_number", "label"]].reset_index(drop=True).copy()
    predictions["predicted_label"] = [names[target] for target in predicted]
    predictions["confidence"] = confidence
    predictions["correct"] = np.asarray(truth) == np.asarray(predicted)
    predictions[
        predictions.predicted_label.isin(unsupported_names)
    ].to_csv(output / "unsupported_class_predictions.csv", index=False)

    false_positive_rows = [
        {"class": name, **values} for name, values in unsupported_false_positives.items()
    ]
    pd.DataFrame(false_positive_rows).to_csv(
        output / "unsupported_class_false_positives.csv", index=False
    )

    matrix = confusion_matrix(truth, predicted, labels=labels)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(output / "confusion_matrix.csv")
    save_confusion_plot(matrix, names, output)
    return report


def save_diagnostic_outputs(
    diagnostic_df: pd.DataFrame,
    truth: list[int],
    predicted: list[int],
    confidence: list[float],
    diagnostic_loss: float,
    class_to_idx: dict[str, int],
    diagnostic_details: dict[str, dict],
    checkpoint_epoch: int,
    output: Path,
) -> None:
    names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    per_class = {}
    for target in sorted(set(truth)):
        positions = [index for index, value in enumerate(truth) if value == target]
        class_predictions = [predicted[index] for index in positions]
        predicted_distribution = pd.Series(
            [names[value] for value in class_predictions], dtype="object"
        ).value_counts()
        correct = sum(value == target for value in class_predictions)
        per_class[names[target]] = {
            "status": "diagnostic_only_same_video",
            "positive_frames": int(len(positions)),
            "correct_frames": int(correct),
            "recall": float(correct / len(positions)),
            "predicted_distribution": {
                str(name): int(count) for name, count in predicted_distribution.items()
            },
            "split": diagnostic_details[names[target]],
        }

    recalls = [values["recall"] for values in per_class.values()]
    report = {
        "evaluation_scope": "diagnostic_only_temporal_holdout_from_same_video",
        "checkpoint_epoch": int(checkpoint_epoch),
        "loss": float(diagnostic_loss),
        "overall_frame_accuracy": float(accuracy_score(truth, predicted)),
        "macro_recall": float(np.mean(recalls)),
        "per_class": per_class,
        "warning": (
            "This probe only checks recognition within the source video. It must not be used "
            "as evidence of generalization to a new patient or video."
        ),
    }
    (output / "diagnostic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions = diagnostic_df[
        ["filename", "video_id", "frame_number", "label"]
    ].reset_index(drop=True).copy()
    predictions["predicted_label"] = [names[target] for target in predicted]
    predictions["confidence"] = confidence
    predictions["correct"] = np.asarray(truth) == np.asarray(predicted)
    predictions.to_csv(output / "diagnostic_predictions.csv", index=False)


def set_trainable_backbone_blocks(model: nn.Module, trainable_blocks: int) -> int:
    """Freeze the backbone, then enable only its final N feature blocks."""
    blocks = list(model.features.children())
    if not 0 <= trainable_blocks <= len(blocks):
        raise ValueError(
            f"trainable-backbone-blocks must be between 0 and {len(blocks)}"
        )
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if trainable_blocks:
        for block in blocks[-trainable_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    return len(blocks)


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    backbone_lr_multiplier: float,
    include_backbone: bool,
) -> torch.optim.Optimizer:
    """Build named parameter groups so a head-only run has no backbone optimizer state."""
    groups = []
    if include_backbone:
        groups.append(
            {
                "params": model.features.parameters(),
                "lr": learning_rate * backbone_lr_multiplier,
                "group_name": "backbone",
            }
        )
    groups.append(
        {
            "params": model.classifier.parameters(),
            "lr": learning_rate,
            "group_name": "classifier",
        }
    )
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images", type=Path,
        default=Path(r"D:\dataset_quazir\labelled_images"),
        help="Extracted class folders (default: D:\\dataset_quazir\\labelled_images)",
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=Path(r"D:\dataset_quazir\metadata.csv"),
        help="Kvasir-Capsule metadata.csv (default: D:\\dataset_quazir\\metadata.csv)",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/regularized_v2"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--workers", type=int, default=0,
        help="DataLoader processes; keep 0 on Windows with 16 GB RAM",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--sampler-power", type=float, default=0.5)
    parser.add_argument("--sampler-max-multiplier", type=float, default=10.0)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backbone-strategy",
        choices=("staged", "head-only"),
        default="staged",
        help=(
            "staged freezes the pretrained backbone first and then fine-tunes its final "
            "blocks; head-only keeps the entire pretrained backbone frozen"
        ),
    )
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--trainable-backbone-blocks", type=int, default=3)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--random-erasing-probability", type=float, default=0.2)
    parser.add_argument(
        "--max-frames-per-video-class",
        type=int,
        default=500,
        help=(
            "Evenly spaced training-frame cap for each (video, class); 0 disables it"
        ),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--imbalance-warning-ratio", type=float, default=10.0)
    parser.add_argument("--diagnostic-fraction", type=float, default=0.25)
    parser.add_argument("--diagnostic-gap-frames", type=int, default=3)
    parser.add_argument("--diagnostic-min-frames", type=int, default=2)
    parser.add_argument("--diagnostic-min-train-frames", type=int, default=4)
    parser.add_argument("--no-rare-diagnostic", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit data and create split reports without loading or training a model",
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("batch-size and epochs must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be positive and weight-decay cannot be negative")
    if not 0 < args.val_size < 1:
        raise ValueError("val-size must be between 0 and 1")
    if not 0 <= args.sampler_power <= 1:
        raise ValueError("sampler-power must be between 0 and 1")
    if args.sampler_max_multiplier < 1:
        raise ValueError("sampler-max-multiplier must be at least 1")
    if args.trainable_backbone_blocks < 0:
        raise ValueError("trainable-backbone-blocks cannot be negative")
    if args.backbone_strategy == "head-only" and args.no_pretrained:
        raise ValueError("head-only requires a pretrained backbone")
    if not 0 < args.backbone_lr_multiplier <= 1:
        raise ValueError("backbone-lr-multiplier must be in (0, 1]")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label-smoothing must be in [0, 1)")
    if not 0 <= args.random_erasing_probability <= 1:
        raise ValueError("random-erasing-probability must be in [0, 1]")
    if args.max_frames_per_video_class < 0:
        raise ValueError("max-frames-per-video-class cannot be negative")
    if not 0 < args.diagnostic_fraction < 1:
        raise ValueError("diagnostic-fraction must be between 0 and 1")
    if args.diagnostic_gap_frames < 0:
        raise ValueError("diagnostic-gap-frames cannot be negative")
    if args.diagnostic_min_frames <= 0 or args.diagnostic_min_train_frames <= 0:
        raise ValueError("diagnostic minimum frame counts must be positive")
    if args.freeze_backbone_epochs < 0 or args.early_stopping_patience < 0:
        raise ValueError("freeze and early-stopping epoch counts cannot be negative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early-stopping-min-delta cannot be negative")

    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    # Combined Kvasir + external metadata has intentionally mixed source-ID and
    # demographic column types. Read the small table in one pass so pandas does
    # not emit chunked-inference warnings for columns unused by training.
    metadata = pd.read_csv(args.metadata, sep=";", low_memory=False)
    raw_metadata_rows = len(metadata)
    required = {"filename", "video_id", "frame_number", "finding_class"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata must contain {sorted(required)}")
    if metadata[list(required)].isna().any().any():
        raise ValueError("Required metadata columns cannot contain missing values")
    metadata["label"] = metadata.finding_class.map(normalize_label)
    duplicate_rows = metadata.duplicated(subset=["filename", "video_id", "frame_number", "label"]).sum()
    if duplicate_rows:
        print(f"Warning: removing {duplicate_rows} duplicate metadata rows")
        metadata = metadata.drop_duplicates(subset=["filename", "video_id", "frame_number", "label"]).copy()
    paths = index_images(args.images)
    metadata["path"] = [paths.get((name, label)) for name, label in zip(metadata.filename, metadata.label)]
    missing = metadata.path.isna().sum()
    if missing:
        print(f"Warning: skipping {missing} metadata rows without an extracted image")
    metadata = metadata.dropna(subset=["path"]).copy()

    labels_per_frame = metadata.groupby(["video_id", "frame_number"]).label.transform("nunique")
    ambiguous_multilabel_rows = metadata[labels_per_frame > 1].copy()
    ambiguous_multilabel_frames = int(
        ambiguous_multilabel_rows[["video_id", "frame_number"]].drop_duplicates().shape[0]
    )
    ambiguous_multilabel_rows.to_csv(
        args.output / "ambiguous_multilabel_rows.csv", index=False
    )
    if ambiguous_multilabel_frames:
        print(
            f"Warning: excluding {ambiguous_multilabel_frames} multi-label frames from the "
            "single-label softmax task; see ambiguous_multilabel_rows.csv"
        )
        metadata = metadata[labels_per_frame == 1].copy()

    data_integrity = {
        "raw_metadata_rows": int(raw_metadata_rows),
        "exact_duplicate_rows_removed": int(duplicate_rows),
        "rows_without_extracted_image_removed": int(missing),
        "ambiguous_multilabel_frames_removed": ambiguous_multilabel_frames,
        "ambiguous_multilabel_rows_removed": int(len(ambiguous_multilabel_rows)),
        "usable_single_label_rows": int(len(metadata)),
        "multilabel_policy": (
            "Frames with more than one correct class are exported for review and excluded, "
            "because duplicate rows with conflicting targets are invalid for softmax cross-entropy."
        ),
    }
    (args.output / "data_integrity.json").write_text(
        json.dumps(data_integrity, indent=2), encoding="utf-8"
    )

    classes = sorted(metadata.label.unique())
    class_to_idx = {name: i for i, name in enumerate(classes)}
    metadata["target"] = metadata.label.map(class_to_idx)

    train_idx, val_idx = choose_group_split(metadata, args.val_size, args.seed)
    group_train_df, val_df = metadata.iloc[train_idx].copy(), metadata.iloc[val_idx].copy()
    overlap = set(group_train_df.video_id) & set(val_df.video_id)
    assert not overlap, "Video leakage detected"

    videos_per_class = metadata.groupby("label").video_id.nunique()
    single_video_classes = sorted(videos_per_class[videos_per_class == 1].index.tolist())
    classes_absent_from_validation = sorted(set(classes) - set(val_df.label.unique()))
    diagnostic_classes = sorted(
        set(single_video_classes) & set(classes_absent_from_validation)
    )
    empty_frame = group_train_df.iloc[0:0].copy()
    if diagnostic_classes and not args.no_rare_diagnostic:
        train_df, diagnostic_df, diagnostic_guard_df, diagnostic_details = (
            create_rare_diagnostic_split(
                group_train_df,
                diagnostic_classes,
                args.diagnostic_fraction,
                args.diagnostic_gap_frames,
                args.diagnostic_min_frames,
                args.diagnostic_min_train_frames,
            )
        )
    else:
        train_df, diagnostic_df, diagnostic_guard_df = group_train_df, empty_frame, empty_frame
        diagnostic_details = {
            name: {
                "status": "disabled" if args.no_rare_diagnostic else "unavailable",
                "reason": "disabled_by_cli" if args.no_rare_diagnostic else "no_eligible_class",
            }
            for name in diagnostic_classes
        }

    train_images_before_temporal_cap = len(train_df)
    train_df, temporal_cap_excluded_df = cap_correlated_training_frames(
        train_df, args.max_frames_per_video_class
    )

    train_df.to_csv(args.output / "train_split.csv", index=False)
    val_df.to_csv(args.output / "val_split.csv", index=False)
    diagnostic_df.to_csv(args.output / "diagnostic_split.csv", index=False)
    diagnostic_guard_df.to_csv(args.output / "diagnostic_guard_excluded.csv", index=False)
    temporal_cap_excluded_df.to_csv(
        args.output / "train_temporal_cap_excluded.csv", index=False
    )
    (args.output / "classes.json").write_text(json.dumps(class_to_idx, indent=2), encoding="utf-8")

    overall_counts = count_by_class(metadata, classes)
    train_counts = count_by_class(train_df, classes)
    val_counts = count_by_class(val_df, classes)
    diagnostic_counts = count_by_class(diagnostic_df, classes)
    train_imbalance = imbalance_ratio(train_counts)
    class_coverage = {
        name: {
            "source_videos": int(videos_per_class[name]),
            "all_frames": overall_counts[name],
            "train_frames": train_counts[name],
            "validation_frames": val_counts[name],
            "diagnostic_frames": diagnostic_counts[name],
            "validation_status": (
                "evaluated_on_unseen_videos"
                if val_counts[name] > 0
                else "sensitivity_not_estimable_no_positive_video"
            ),
        }
        for name in classes
    }
    split_summary = {
        "train_images": len(train_df),
        "train_images_before_temporal_cap": train_images_before_temporal_cap,
        "train_images_excluded_by_temporal_cap": len(temporal_cap_excluded_df),
        "max_frames_per_video_class": int(args.max_frames_per_video_class),
        "val_images": len(val_df),
        "diagnostic_images": len(diagnostic_df),
        "diagnostic_guard_excluded_rows": len(diagnostic_guard_df),
        "train_videos": int(train_df.video_id.nunique()), "val_videos": int(val_df.video_id.nunique()),
        "classes_absent_from_validation": classes_absent_from_validation,
        "single_video_classes": single_video_classes,
        "class_coverage": class_coverage,
        "diagnostic_split": diagnostic_details,
        "note": (
            "Classes sourced from only one video cannot occur in both leak-free partitions. "
            "The temporal diagnostic holdout is same-video and is never used for model selection."
        ),
    }
    (args.output / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    save_class_distribution(train_df, val_df, diagnostic_df, args.output)

    class_balance = {
        "overall_counts": overall_counts,
        "train_counts": train_counts,
        "validation_counts": val_counts,
        "diagnostic_only_counts": diagnostic_counts,
        "overall_max_to_min_ratio": imbalance_ratio(overall_counts),
        "train_max_to_min_ratio": train_imbalance,
        "warning_threshold": float(args.imbalance_warning_ratio),
        "severe_imbalance": bool(
            train_imbalance is not None and train_imbalance >= args.imbalance_warning_ratio
        ),
        "sampler": {
            "strategy": "capped_power_weighting",
            "power": float(args.sampler_power),
            "max_per_sample_multiplier": float(args.sampler_max_multiplier),
            "note": (
                "power=0.5 approximates square-root rebalancing; the cap prevents a few "
                "rare frames from being repeated hundreds of times per epoch."
            ),
        },
    }
    if class_balance["severe_imbalance"]:
        print(
            f"Warning: severe train class imbalance, max/min={train_imbalance:.1f}; "
            "using capped square-root sampling"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, diagnostic_loader, sampler_details = build_loaders(
        train_df,
        val_df,
        diagnostic_df,
        args.batch_size,
        args.workers,
        args.sampler_power,
        args.sampler_max_multiplier,
        args.random_erasing_probability,
    )
    names_by_target = {target: name for name, target in class_to_idx.items()}
    class_balance["sampler"]["per_class_expected_sampling"] = {
        names_by_target[target]: details for target, details in sampler_details.items()
    }
    (args.output / "class_balance.json").write_text(
        json.dumps(class_balance, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(
            f"Dry run complete: train={len(train_df)}; val={len(val_df)}; "
            f"diagnostic={len(diagnostic_df)}; output={args.output}"
        )
        return

    pretrained = None if args.no_pretrained else EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=pretrained)
    model.classifier[0] = nn.Dropout(p=args.dropout, inplace=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    permanently_frozen_backbone = args.backbone_strategy == "head-only"
    freeze_backbone_epochs = 0 if args.no_pretrained else args.freeze_backbone_epochs
    total_backbone_blocks = len(list(model.features.children()))
    if args.trainable_backbone_blocks > total_backbone_blocks:
        raise ValueError(
            f"trainable-backbone-blocks cannot exceed {total_backbone_blocks} for EfficientNet-B0"
        )
    if permanently_frozen_backbone:
        trainable_backbone_blocks = 0
    elif args.no_pretrained:
        trainable_backbone_blocks = total_backbone_blocks
    elif freeze_backbone_epochs > 0:
        trainable_backbone_blocks = 0
    else:
        trainable_backbone_blocks = args.trainable_backbone_blocks
    set_trainable_backbone_blocks(model, trainable_backbone_blocks)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    effective_backbone_lr_multiplier = (
        1.0 if args.no_pretrained else args.backbone_lr_multiplier
    )
    optimizer = build_optimizer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_multiplier=effective_backbone_lr_multiplier,
        include_backbone=not permanently_frozen_backbone,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    training_config = {
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "backbone_strategy": args.backbone_strategy,
        "backbone_frozen_for_entire_run": bool(permanently_frozen_backbone),
        "classifier_learning_rate": float(args.lr),
        "backbone_learning_rate": float(
            0.0
            if permanently_frozen_backbone
            else args.lr * effective_backbone_lr_multiplier
        ),
        "backbone_lr_multiplier": float(
            0.0 if permanently_frozen_backbone else effective_backbone_lr_multiplier
        ),
        "freeze_backbone_epochs": int(
            args.epochs if permanently_frozen_backbone else freeze_backbone_epochs
        ),
        "trainable_backbone_blocks_after_freeze": int(
            0
            if permanently_frozen_backbone
            else total_backbone_blocks
            if args.no_pretrained
            else args.trainable_backbone_blocks
        ),
        "total_backbone_blocks": int(total_backbone_blocks),
        "dropout": float(args.dropout),
        "label_smoothing": float(args.label_smoothing),
        "weight_decay": float(args.weight_decay),
        "random_erasing_probability": float(args.random_erasing_probability),
        "max_frames_per_video_class": int(args.max_frames_per_video_class),
        "sampler_power": float(args.sampler_power),
        "sampler_max_multiplier": float(args.sampler_max_multiplier),
    }
    (args.output / "training_config.json").write_text(
        json.dumps(training_config, indent=2), encoding="utf-8"
    )

    supported_targets = sorted(int(value) for value in val_df.target.unique())
    unsupported_names = [
        name for name, target in class_to_idx.items() if target not in supported_targets
    ]
    print(
        f"Device={device}; batch={args.batch_size}; train={len(train_df)}; val={len(val_df)}; "
        f"diagnostic={len(diagnostic_df)}; classes={len(classes)}"
    )
    if permanently_frozen_backbone:
        print(
            f"Backbone strategy=head-only; all {total_backbone_blocks} backbone blocks "
            "remain frozen for the entire run"
        )
    else:
        print(
            f"Backbone strategy=staged; freeze_epochs={freeze_backbone_epochs}; "
            f"final_trainable_blocks={training_config['trainable_backbone_blocks_after_freeze']}"
        )
    print(f"Validation sensitivity not estimable for: {unsupported_names}")
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        if (
            not permanently_frozen_backbone
            and epoch == freeze_backbone_epochs + 1
            and freeze_backbone_epochs > 0
        ):
            set_trainable_backbone_blocks(model, args.trainable_backbone_blocks)
            print(
                f"Epoch {epoch}: unfroze final {args.trainable_backbone_blocks}/"
                f"{total_backbone_blocks} backbone blocks"
            )

        train_loss, train_f1, _, _, _ = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
            scaler,
            metric_labels=list(range(len(classes))),
        )
        val_loss, val_f1, truth, pred, confidence = run_epoch(
            model, val_loader, loss_fn, device, metric_labels=supported_targets
        )
        current_lrs = {
            group["group_name"]: group["lr"] for group in optimizer.param_groups
        }
        backbone_lr = current_lrs.get("backbone", 0.0)
        classifier_lr = current_lrs["classifier"]
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": classifier_lr,
            "backbone_learning_rate": backbone_lr,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_supported_macro_f1": val_f1,
        }
        history.append(row)
        print(json.dumps(row))
        if val_f1 > best_f1 + args.early_stopping_min_delta:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": class_to_idx,
                    "epoch": epoch,
                    "val_supported_macro_f1": val_f1,
                    "unsupported_validation_classes": unsupported_names,
                    "training_config": training_config,
                },
                args.output / "best.pt",
            )
            save_validation_outputs(
                val_df, truth, pred, confidence, class_to_idx, args.output
            )
        else:
            epochs_without_improvement += 1
        pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
        save_history_plot(history, args.output)
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; no supported macro-F1 improvement "
                f"for {epochs_without_improvement} epochs"
            )
            break

    checkpoint = torch.load(args.output / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    if diagnostic_loader is not None:
        diagnostic_targets = sorted(int(value) for value in diagnostic_df.target.unique())
        diagnostic_loss, _, diagnostic_truth, diagnostic_pred, diagnostic_confidence = run_epoch(
            model,
            diagnostic_loader,
            loss_fn,
            device,
            metric_labels=diagnostic_targets,
        )
        save_diagnostic_outputs(
            diagnostic_df,
            diagnostic_truth,
            diagnostic_pred,
            diagnostic_confidence,
            diagnostic_loss,
            class_to_idx,
            diagnostic_details,
            best_epoch,
            args.output,
        )
    else:
        (args.output / "diagnostic_report.json").write_text(
            json.dumps(
                {
                    "evaluation_scope": "diagnostic_only_temporal_holdout_from_same_video",
                    "status": "unavailable",
                    "details": diagnostic_details,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        f"Best supported-class validation macro-F1: {best_f1:.4f} at epoch {best_epoch}; "
        f"checkpoint: {args.output / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
