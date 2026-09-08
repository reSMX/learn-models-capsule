"""Diagnostic-only threshold sweep for a trained Galar multi-label checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GalarMultiLabelDataset
from train_common import (
    build_model,
    build_transforms,
    load_task_config,
    resolve_device,
    seed_everything,
    seed_loader_worker,
    sha256_file,
    validate_image_cache,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep sigmoid thresholds on unseen-study validation. Results are "
            "diagnostic-only and must not be used for checkpoint selection."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-cache-root", type=Path)
    parser.add_argument("--galar-root", type=Path)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "metadata",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.image_cache_root is None and args.galar_root is None:
        parser.error("provide --image-cache-root or --galar-root")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers cannot be negative")
    if not 0.0 < args.threshold_start <= args.threshold_end < 1.0:
        parser.error("threshold range must be ordered within (0, 1)")
    if args.threshold_step <= 0:
        parser.error("threshold-step must be positive")
    return args


class ThresholdSweep:
    def __init__(
        self, thresholds: torch.Tensor, classes: list[str], device: torch.device
    ) -> None:
        self.thresholds = thresholds.to(device)
        self.classes = classes
        shape = (len(thresholds), len(classes))
        self.tp = torch.zeros(shape, dtype=torch.int64, device=device)
        self.fp = torch.zeros(shape, dtype=torch.int64, device=device)
        self.fn = torch.zeros(shape, dtype=torch.int64, device=device)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probabilities = torch.sigmoid(logits.detach())
        predicted = probabilities.unsqueeze(0) >= self.thresholds[:, None, None]
        truth = (targets.detach() >= 0.5).unsqueeze(0)
        self.tp += (predicted & truth).sum(dim=1)
        self.fp += (predicted & ~truth).sum(dim=1)
        self.fn += (~predicted & truth).sum(dim=1)

    def compute(self) -> dict[str, Any]:
        tp = self.tp.cpu().to(torch.float64)
        fp = self.fp.cpu().to(torch.float64)
        fn = self.fn.cpu().to(torch.float64)
        denominator = 2 * tp + fp + fn
        f1 = torch.where(denominator > 0, 2 * tp / denominator, 0.0)
        supported = (tp + fn)[0] > 0
        macro_f1 = f1[:, supported].mean(dim=1)
        threshold_values = self.thresholds.cpu().tolist()
        best_global_index = int(macro_f1.argmax())

        per_class: dict[str, dict[str, float | int]] = {}
        for class_index, label in enumerate(self.classes):
            best_index = int(f1[:, class_index].argmax())
            per_class[label] = {
                "threshold": float(threshold_values[best_index]),
                "f1": float(f1[best_index, class_index]),
                "support": int((tp + fn)[0, class_index]),
            }

        return {
            "global_thresholds": [
                {
                    "threshold": float(threshold),
                    "supported_macro_f1": float(macro_f1[index]),
                }
                for index, threshold in enumerate(threshold_values)
            ],
            "best_global_threshold": float(threshold_values[best_global_index]),
            "best_global_supported_macro_f1": float(macro_f1[best_global_index]),
            "per_class_best": per_class,
        }


def threshold_grid(start: float, end: float, step: float) -> torch.Tensor:
    values: list[float] = []
    current = start
    while current <= end + step / 100:
        values.append(round(current, 10))
        current += step
    if 0.5 not in values:
        values.append(0.5)
        values.sort()
    return torch.tensor(values, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"Unsupported Galar checkpoint format: {checkpoint_path}")
    task = checkpoint.get("task")
    if task not in {"anatomy", "pathology"}:
        raise ValueError(f"Unsupported checkpoint task: {task!r}")

    metadata_dir = args.metadata_dir.resolve()
    config = load_task_config(metadata_dir, task)
    classes = list(config["classes"])
    if list(checkpoint.get("classes", ())) != classes:
        raise ValueError("Checkpoint class order does not match current task metadata")
    validation_metadata = metadata_dir / config["validation_file"]
    train_metadata = metadata_dir / config["train_file"]
    image_cache_root = (
        args.image_cache_root.resolve() if args.image_cache_root is not None else None
    )
    if image_cache_root is not None:
        validate_image_cache(image_cache_root, (train_metadata, validation_metadata))

    _, validation_transform = build_transforms()
    dataset = GalarMultiLabelDataset(
        validation_metadata,
        args.galar_root,
        classes,
        validation_transform,
        image_cache_root=image_cache_root,
    )
    device = resolve_device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_loader_worker,
        pin_memory=device.type == "cuda",
    )
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    thresholds = threshold_grid(
        args.threshold_start, args.threshold_end, args.threshold_step
    )
    sweep = ThresholdSweep(thresholds, classes, device)
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="threshold audit"):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                sweep.update(model(images), targets)

    output = args.output or checkpoint_path.parent / "threshold_diagnostics.json"
    output = output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")
    result = {
        "status": "diagnostic_only",
        "warning": (
            "Thresholds were optimized on the same unseen-study validation set; "
            "do not use these results for checkpoint selection or as an unbiased test."
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "task": task,
        "classes": classes,
        "validation_frames": len(dataset),
        **sweep.compute(),
    }
    write_json_atomic(output, result)
    print(json.dumps(result, indent=2))
    print(f"Diagnostic thresholds written to: {output}")


if __name__ == "__main__":
    main()
