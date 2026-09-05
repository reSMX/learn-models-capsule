"""Shared EfficientNet-B0 training loop for Galar multi-label tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import v2
from tqdm import tqdm

from dataset import GalarMultiLabelDataset


IMAGE_SIZE = 224


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_loader_worker(worker_id: int) -> None:
    """Seed a spawned Windows worker and avoid CPU thread oversubscription."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)


def parse_args(task: str, default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train the Galar {task} multi-label EfficientNet-B0 baseline."
    )
    parser.add_argument(
        "--galar-root",
        type=Path,
        help="Original Galar PNG root; optional when --image-cache-root is used",
    )
    parser.add_argument(
        "--image-cache-root",
        type=Path,
        help="SSD cache produced by prepare_image_cache.py (preferred over original PNGs)",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "metadata",
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker processes (2 is the tested Windows default; use 0 to debug)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched by each worker; used only when workers > 0",
    )
    parser.add_argument(
        "--loss-display-interval",
        type=int,
        default=50,
        help="Synchronise and refresh live batch loss every N batches (0 disables it)",
    )
    parser.add_argument(
        "--ram-buffer-gb",
        type=float,
        default=2.0,
        help=(
            "Total RAM for two alternating decoded-tensor buffers; "
            "0 disables buffering"
        ),
    )
    parser.add_argument(
        "--read-block-size",
        type=int,
        default=512,
        help=(
            "Contiguous metadata rows read per shuffled locality block when "
            "RAM buffering is enabled"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument(
        "--unfreeze-stages-per-epoch",
        type=int,
        default=2,
        help=(
            "EfficientNet feature stages unfrozen from the output backwards each "
            "epoch after the frozen warm-up; 0 restores one-step full unfreezing"
        ),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=100.0,
        help="Cap for BCE positive weights derived from training prevalence",
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one tiny train/validation pass without downloads or output files",
    )
    parser.add_argument("--dry-run-samples", type=int, default=4)
    args = parser.parse_args()

    if args.galar_root is None and args.image_cache_root is None:
        parser.error("provide --image-cache-root or --galar-root")

    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers cannot be negative")
    if args.prefetch_factor < 1 or args.loss_display_interval < 0:
        parser.error("prefetch-factor must be positive; loss-display-interval cannot be negative")
    if args.ram_buffer_gb < 0 or args.read_block_size < 1:
        parser.error("ram-buffer-gb cannot be negative; read-block-size must be positive")
    if args.freeze_backbone_epochs < 0 or args.early_stopping_patience < 0:
        parser.error("freeze and early-stopping epoch counts cannot be negative")
    if args.unfreeze_stages_per_epoch < 0:
        parser.error("unfreeze-stages-per-epoch cannot be negative")
    if not 0.0 < args.threshold < 1.0:
        parser.error("threshold must be between 0 and 1")
    if args.max_pos_weight < 1.0 or args.gradient_clip < 0:
        parser.error("max-pos-weight must be >= 1 and gradient-clip cannot be negative")
    if args.dry_run_samples < 1:
        parser.error("dry-run-samples must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_image_cache(
    cache_root: Path, metadata_paths: tuple[Path, Path]
) -> dict[str, Any]:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Complete SSD cache not found: {manifest_path}. "
            "Run prepare_image_cache.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "galar_jpeg_shards_v1" or manifest.get("status") != "complete":
        raise ValueError(f"Unsupported or incomplete SSD cache: {manifest_path}")
    recorded = manifest.get("metadata_files")
    if not isinstance(recorded, dict):
        raise ValueError(f"SSD cache has no metadata fingerprints: {manifest_path}")
    for path in metadata_paths:
        expected = recorded.get(path.name)
        if not isinstance(expected, dict):
            raise ValueError(f"SSD cache was not built for metadata file: {path.name}")
        if path.stat().st_size != int(expected["bytes"]) or sha256_file(path) != expected["sha256"]:
            raise ValueError(
                f"Metadata changed since the SSD cache was built: {path}. "
                "Rebuild the image cache before training."
            )
    settings = manifest.get("settings", {})
    if int(settings.get("image_size", 0)) < 256:
        raise ValueError("SSD cache images are smaller than the required 256 pixels")
    return manifest


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_task_config(metadata_dir: Path, task: str) -> dict[str, Any]:
    path = metadata_dir / f"{task}_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Task configuration not found: {path}. Run prepare_metadata.py first."
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("task") != task or config.get("problem_type") != "multi_label":
        raise ValueError(f"Unexpected task configuration in {path}")
    classes = config.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"No classes recorded in {path}")

    split_path = metadata_dir / str(config["split_file"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_studies = set(split["train_studies"])
    validation_studies = set(split["validation_studies"])
    overlap = train_studies & validation_studies
    if overlap:
        raise ValueError(f"Study leakage in {split_path}: {sorted(overlap)}")
    if not train_studies or not validation_studies:
        raise ValueError(f"Empty train or validation study set in {split_path}")
    return config


def build_transforms() -> tuple[v2.Compose, v2.Compose]:
    weights = EfficientNet_B0_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.75, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(25),
            v2.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.02),
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.10
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.5, 2.0)),
        ]
    )
    validation_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(IMAGE_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, validation_transform


def build_model(class_count: int, pretrained: bool) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, class_count)
    return model


def set_trainable_backbone_stages(model: nn.Module, trainable_stages: int) -> int:
    """Unfreeze the requested number of EfficientNet stages from the output back."""

    stages = list(model.features.children())
    trainable_stages = min(max(0, trainable_stages), len(stages))
    first_trainable = len(stages) - trainable_stages
    for index, stage in enumerate(stages):
        for parameter in stage.parameters():
            parameter.requires_grad = index >= first_trainable
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return len(stages)


def backbone_stages_for_epoch(
    epoch: int,
    freeze_epochs: int,
    stages_per_epoch: int,
    total_stages: int,
) -> int:
    if epoch <= freeze_epochs:
        return 0
    if stages_per_epoch == 0:
        return total_stages
    return min(total_stages, (epoch - freeze_epochs) * stages_per_epoch)


def build_optimizer(model: nn.Module, args: argparse.Namespace):
    # Keep both parameter groups present from the start. Frozen parameters have
    # no gradients, and become trainable without resetting AdamW or the scheduler.
    groups = [
        {"params": model.features.parameters(), "lr": args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": args.head_lr},
    ]
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def positive_weights(config: dict[str, Any], classes: list[str], cap: float) -> torch.Tensor:
    total = int(config["train"]["frames"])
    counts = config["train"]["class_frame_counts"]
    weights: list[float] = []
    for label in classes:
        positive = int(counts[label])
        if positive <= 0:
            raise ValueError(f"Training split has no positive frames for {label!r}")
        ratio = (total - positive) / positive
        weights.append(min(cap, max(1.0, ratio)))
    return torch.tensor(weights, dtype=torch.float32)


class LocalityBlockSampler(Sampler[int]):
    """Shuffle contiguous file blocks while preserving sequential reads inside each block."""

    def __init__(
        self, sample_count: int, block_size: int, generator: torch.Generator
    ) -> None:
        self.sample_count = sample_count
        self.block_size = block_size
        self.generator = generator

    def __len__(self) -> int:
        return self.sample_count

    def __iter__(self) -> Iterator[int]:
        block_count = math.ceil(self.sample_count / self.block_size)
        order = torch.randperm(block_count, generator=self.generator).tolist()
        for block_index in order:
            start = block_index * self.block_size
            stop = min(start + self.block_size, self.sample_count)
            yield from range(start, stop)


@dataclass
class RamChunk:
    images: torch.Tensor
    targets: torch.Tensor

    @property
    def sample_count(self) -> int:
        return len(self.targets)


def load_ram_chunk(
    source: Iterator[tuple[torch.Tensor, torch.Tensor]],
    batch_limit: int,
    batch_size: int,
) -> RamChunk | None:
    """Copy already decoded DataLoader batches into one ordinary-RAM slot."""

    images_cache: torch.Tensor | None = None
    targets_cache: torch.Tensor | None = None
    offset = 0
    for _ in range(batch_limit):
        try:
            images, targets = next(source)
        except StopIteration:
            break
        if images_cache is None:
            capacity = batch_limit * batch_size
            images_cache = torch.empty(
                (capacity, *images.shape[1:]), dtype=images.dtype
            )
            targets_cache = torch.empty(
                (capacity, *targets.shape[1:]), dtype=targets.dtype
            )
        count = len(images)
        images_cache[offset : offset + count].copy_(images)
        assert targets_cache is not None
        targets_cache[offset : offset + count].copy_(targets)
        offset += count
    if images_cache is None or targets_cache is None:
        return None
    return RamChunk(images_cache[:offset], targets_cache[:offset])


class DoubleBufferedLoader:
    """Overlap GPU work with block-local disk reads into two bounded RAM slots."""

    def __init__(
        self,
        loader: DataLoader,
        batches_per_slot: int,
        batch_size: int,
        shuffle: bool,
        pin_output: bool,
        generator: torch.Generator,
        name: str,
    ) -> None:
        self.loader = loader
        self.batches_per_slot = batches_per_slot
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_output = pin_output
        self.generator = generator
        self.name = name

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        source = iter(self.loader)
        remaining_batches = len(self.loader)
        first_batches = min(self.batches_per_slot, remaining_batches)
        remaining_batches -= first_batches
        tqdm.write(
            f"{self.name} RAM warm-up: loading {first_batches} decoded batches "
            "before GPU processing..."
        )
        warmup_started = time.perf_counter()
        current = load_ram_chunk(source, first_batches, self.batch_size)
        if current is None:
            return
        tqdm.write(
            f"{self.name} RAM warm-up ready: {current.sample_count} frames in "
            f"{time.perf_counter() - warmup_started:.1f}s"
        )

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="galar-ram-buffer") as pool:
            next_batches = min(self.batches_per_slot, remaining_batches)
            remaining_batches -= next_batches
            future = (
                pool.submit(load_ram_chunk, source, next_batches, self.batch_size)
                if next_batches
                else None
            )
            while current is not None:
                if self.shuffle:
                    order: torch.Tensor | slice = torch.randperm(
                        current.sample_count, generator=self.generator
                    )
                else:
                    order = slice(None)

                for start in range(0, current.sample_count, self.batch_size):
                    stop = min(start + self.batch_size, current.sample_count)
                    indices = (
                        order[start:stop]
                        if isinstance(order, torch.Tensor)
                        else slice(start, stop)
                    )
                    images = current.images[indices]
                    targets = current.targets[indices]
                    if self.pin_output:
                        images = images.pin_memory()
                        targets = targets.pin_memory()
                    yield images, targets

                refill_started = time.perf_counter()
                refill_was_ready = future is None or future.done()
                current = future.result() if future is not None else None
                refill_wait = time.perf_counter() - refill_started
                if not refill_was_ready and refill_wait >= 1.0:
                    tqdm.write(
                        f"{self.name} RAM buffer waited {refill_wait:.1f}s for the "
                        "background slot; storage/decode throughput is below GPU demand"
                    )
                next_batches = min(self.batches_per_slot, remaining_batches)
                remaining_batches -= next_batches
                future = (
                    pool.submit(load_ram_chunk, source, next_batches, self.batch_size)
                    if current is not None and next_batches
                    else None
                )


def ram_batches_per_slot(
    ram_buffer_gb: float, batch_size: int, class_count: int
) -> int:
    """Split the configured memory approximately evenly across two RAM slots."""

    bytes_per_sample = IMAGE_SIZE * IMAGE_SIZE * 3 * 4 + class_count * 4
    bytes_per_batch = bytes_per_sample * batch_size
    total_bytes = int(ram_buffer_gb * 1024**3)
    return max(1, total_bytes // 2 // bytes_per_batch)


class StreamingMultiLabelMetrics:
    def __init__(
        self, classes: list[str], threshold: float, device: torch.device
    ) -> None:
        self.classes = classes
        self.threshold = threshold
        size = len(classes)
        self.tp = torch.zeros(size, dtype=torch.int64, device=device)
        self.fp = torch.zeros(size, dtype=torch.int64, device=device)
        self.fn = torch.zeros(size, dtype=torch.int64, device=device)
        self.tn = torch.zeros(size, dtype=torch.int64, device=device)
        self.exact = torch.zeros((), dtype=torch.int64, device=device)
        self.samples = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predicted = torch.sigmoid(logits.detach()) >= self.threshold
        truth = targets.detach() >= 0.5
        self.tp += (predicted & truth).sum(dim=0)
        self.fp += (predicted & ~truth).sum(dim=0)
        self.fn += (~predicted & truth).sum(dim=0)
        self.tn += (~predicted & ~truth).sum(dim=0)
        self.exact += (predicted == truth).all(dim=1).sum()
        self.samples += len(truth)

    def compute(self) -> dict[str, Any]:
        # One device synchronisation per phase instead of four transfers per batch.
        tp = self.tp.cpu().to(torch.float64)
        fp = self.fp.cpu().to(torch.float64)
        fn = self.fn.cpu().to(torch.float64)
        tn = self.tn.cpu().to(torch.float64)
        exact = int(self.exact.cpu())
        precision = tp / (tp + fp).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        support = tp + fn
        supported = support > 0
        macro_f1 = float(f1[supported].mean()) if supported.any() else 0.0
        total_tp = tp.sum()
        micro_precision = total_tp / (total_tp + fp.sum()).clamp_min(1)
        micro_recall = total_tp / (total_tp + fn.sum()).clamp_min(1)
        micro_f1 = (
            2 * micro_precision * micro_recall
            / (micro_precision + micro_recall).clamp_min(1e-12)
        )
        per_class = {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "predicted_positive": int(tp[index] + fp[index]),
                "tp": int(tp[index]),
                "fp": int(fp[index]),
                "fn": int(fn[index]),
                "tn": int(tn[index]),
            }
            for index, label in enumerate(self.classes)
        }
        return {
            "threshold": self.threshold,
            "supported_macro_f1": macro_f1,
            "micro_f1": float(micro_f1),
            "exact_match": exact / max(1, self.samples),
            "samples": self.samples,
            "unsupported_classes": [
                label for index, label in enumerate(self.classes) if not supported[index]
            ],
            "per_class": per_class,
        }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float,
    classes: list[str],
    optimizer=None,
    scaler=None,
    amp_enabled: bool = False,
    gradient_clip: float = 0.0,
    trainable_backbone_stages: int | None = None,
    loss_display_interval: int = 50,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    if training and trainable_backbone_stages is not None:
        stages = list(model.features.children())
        frozen_stage_count = len(stages) - trainable_backbone_stages
        # Keep BatchNorm statistics fixed in stages whose weights are frozen.
        for stage in stages[:frozen_stage_count]:
            stage.eval()
    metrics = StreamingMultiLabelMetrics(classes, threshold, device)
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_samples = 0

    progress = tqdm(loader, desc="train" if training else "validation", leave=False)
    for step, (images, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = loss_fn(logits, targets)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()

        batch_size = len(targets)
        total_loss += loss.detach().to(torch.float64) * batch_size
        total_samples += batch_size
        metrics.update(logits, targets)
        if loss_display_interval > 0 and (
            step == 1 or step % loss_display_interval == 0 or step == len(loader)
        ):
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

    result = metrics.compute()
    result["loss"] = float(total_loss.cpu()) / max(1, total_samples)
    return result


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_history(output: Path, history: list[dict[str, Any]]) -> None:
    write_json_atomic(output / "history.json", history)
    fields = (
        "epoch",
        "backbone_frozen",
        "trainable_backbone_stages",
        "train_loss",
        "train_supported_macro_f1",
        "train_micro_f1",
        "validation_loss",
        "validation_supported_macro_f1",
        "validation_micro_f1",
    )
    temporary = output / "history.csv.part"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in history:
            writer.writerow(
                {
                    "epoch": item["epoch"],
                    "backbone_frozen": item["backbone_frozen"],
                    "trainable_backbone_stages": item[
                        "trainable_backbone_stages"
                    ],
                    "train_loss": item["train"]["loss"],
                    "train_supported_macro_f1": item["train"]["supported_macro_f1"],
                    "train_micro_f1": item["train"]["micro_f1"],
                    "validation_loss": item["validation"]["loss"],
                    "validation_supported_macro_f1": item["validation"][
                        "supported_macro_f1"
                    ],
                    "validation_micro_f1": item["validation"]["micro_f1"],
                }
            )
    temporary.replace(output / "history.csv")


def run_training(task: str, default_output: Path) -> None:
    args = parse_args(task, default_output)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    metadata_dir = args.metadata_dir.resolve()
    config = load_task_config(metadata_dir, task)
    classes = list(config["classes"])
    train_transform, validation_transform = build_transforms()
    sample_limit = args.dry_run_samples if args.dry_run else None
    train_metadata = metadata_dir / config["train_file"]
    validation_metadata = metadata_dir / config["validation_file"]
    image_cache_root = (
        args.image_cache_root.resolve() if args.image_cache_root is not None else None
    )
    if image_cache_root is not None:
        cache_manifest = validate_image_cache(
            image_cache_root, (train_metadata, validation_metadata)
        )
    else:
        cache_manifest = None

    train_dataset = GalarMultiLabelDataset(
        train_metadata,
        args.galar_root,
        classes,
        train_transform,
        max_samples=sample_limit,
        image_cache_root=image_cache_root,
    )
    validation_dataset = GalarMultiLabelDataset(
        validation_metadata,
        args.galar_root,
        classes,
        validation_transform,
        max_samples=sample_limit,
        image_cache_root=image_cache_root,
    )
    ram_buffer_enabled = args.ram_buffer_gb > 0
    train_generator = torch.Generator().manual_seed(args.seed)
    validation_generator = torch.Generator().manual_seed(args.seed + 1)
    block_generator = torch.Generator().manual_seed(args.seed + 2)
    ram_shuffle_generator = torch.Generator().manual_seed(args.seed + 3)
    worker_options = {
        "num_workers": args.workers,
        # A RAM-buffered loader pins only the batch being transferred, rather
        # than page-locking the complete multi-gigabyte cache.
        "pin_memory": device.type == "cuda" and not ram_buffer_enabled,
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_loader_worker,
    }
    if args.workers > 0:
        worker_options["prefetch_factor"] = args.prefetch_factor
    effective_train_batch_size = min(args.batch_size, len(train_dataset))
    train_sampler = (
        LocalityBlockSampler(
            len(train_dataset), args.read_block_size, block_generator
        )
        if ram_buffer_enabled
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=not ram_buffer_enabled,
        sampler=train_sampler,
        generator=train_generator,
        batch_size=effective_train_batch_size,
        **worker_options,
    )
    effective_validation_batch_size = min(args.batch_size, len(validation_dataset))
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        generator=validation_generator,
        batch_size=effective_validation_batch_size,
        **worker_options,
    )
    if ram_buffer_enabled:
        batches_per_slot = ram_batches_per_slot(
            args.ram_buffer_gb, args.batch_size, len(classes)
        )
        train_phase_loader = DoubleBufferedLoader(
            train_loader,
            batches_per_slot=batches_per_slot,
            batch_size=effective_train_batch_size,
            shuffle=True,
            pin_output=device.type == "cuda",
            generator=ram_shuffle_generator,
            name="train",
        )
        validation_phase_loader = DoubleBufferedLoader(
            validation_loader,
            batches_per_slot=batches_per_slot,
            batch_size=effective_validation_batch_size,
            shuffle=False,
            pin_output=device.type == "cuda",
            generator=torch.Generator().manual_seed(args.seed + 4),
            name="validation",
        )
        estimated_bytes = (
            2
            * batches_per_slot
            * args.batch_size
            * (IMAGE_SIZE * IMAGE_SIZE * 3 * 4 + len(classes) * 4)
        )
        print(
            f"RAM double buffer: 2 x {batches_per_slot} batches "
            f"(~{estimated_bytes / 1024**3:.2f} GiB); "
            f"locality read block={args.read_block_size} frames"
        )
    else:
        train_phase_loader = train_loader
        validation_phase_loader = validation_loader

    pretrained = not args.no_pretrained and not args.dry_run
    model = build_model(len(classes), pretrained=pretrained).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    total_backbone_stages = set_trainable_backbone_stages(model, 0)
    trainable_backbone_stages = 0
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    pos_weight = positive_weights(config, classes, args.max_pos_weight).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(
        f"Task={task}; classes={len(classes)}; device={device}; "
        f"train={len(train_dataset)}; validation={len(validation_dataset)}; "
        f"batch_size={args.batch_size}; workers={args.workers}; "
        f"AMP={amp_enabled}; pretrained={pretrained}"
    )
    if cache_manifest is not None:
        settings = cache_manifest["settings"]
        print(
            f"Image source=SSD JPEG shards at {image_cache_root}; "
            f"size={settings['image_size']}; quality={settings['jpeg_quality']}; "
            f"subsampling={settings['jpeg_subsampling']}"
        )
    else:
        print(f"Image source=original PNG files at {args.galar_root.resolve()}")
    print("Classes:", ", ".join(classes))
    print(
        "BCE pos_weight:",
        ", ".join(
            f"{label}={weight:.3g}"
            for label, weight in zip(classes, pos_weight.cpu().tolist(), strict=True)
        ),
    )

    if args.dry_run:
        train_metrics = run_epoch(
            model,
            train_phase_loader,
            loss_fn,
            device,
            args.threshold,
            classes,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_clip=args.gradient_clip,
            trainable_backbone_stages=trainable_backbone_stages,
            loss_display_interval=args.loss_display_interval,
        )
        validation_metrics = run_epoch(
            model,
            validation_phase_loader,
            loss_fn,
            device,
            args.threshold,
            classes,
            amp_enabled=amp_enabled,
            loss_display_interval=args.loss_display_interval,
        )
        print(
            f"Dry run passed: train loss={train_metrics['loss']:.4f}; "
            f"validation loss={validation_metrics['loss']:.4f}; no files written."
        )
        return

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_config = {
        "task": task,
        "problem_type": "multi_label",
        "classes": classes,
        "activation": "sigmoid",
        "loss": "BCEWithLogitsLoss",
        "validation_threshold": args.threshold,
        "model_selection_metric": "validation_supported_macro_f1_at_fixed_threshold",
        "backbone_stage_count": total_backbone_stages,
        "pos_weight": {
            label: float(value)
            for label, value in zip(classes, pos_weight.cpu(), strict=True)
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    write_json_atomic(output / "run_config.json", run_config)

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        requested_stages = backbone_stages_for_epoch(
            epoch,
            args.freeze_backbone_epochs,
            args.unfreeze_stages_per_epoch,
            total_backbone_stages,
        )
        if requested_stages != trainable_backbone_stages:
            trainable_backbone_stages = requested_stages
            set_trainable_backbone_stages(model, trainable_backbone_stages)
            print(
                f"Epoch {epoch}: trainable EfficientNet-B0 backbone stages "
                f"{trainable_backbone_stages}/{total_backbone_stages} "
                "(from output backwards)"
            )
        backbone_frozen = trainable_backbone_stages == 0

        train_metrics = run_epoch(
            model,
            train_phase_loader,
            loss_fn,
            device,
            args.threshold,
            classes,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_clip=args.gradient_clip,
            trainable_backbone_stages=trainable_backbone_stages,
            loss_display_interval=args.loss_display_interval,
        )
        validation_metrics = run_epoch(
            model,
            validation_phase_loader,
            loss_fn,
            device,
            args.threshold,
            classes,
            amp_enabled=amp_enabled,
            loss_display_interval=args.loss_display_interval,
        )
        epoch_result = {
            "epoch": epoch,
            "backbone_frozen": backbone_frozen,
            "trainable_backbone_stages": trainable_backbone_stages,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_result)
        write_history(output, history)
        validation_f1 = float(validation_metrics["supported_macro_f1"])
        scheduler.step(validation_f1)
        print(
            f"Epoch {epoch}/{args.epochs}: train loss={train_metrics['loss']:.4f}, "
            f"val loss={validation_metrics['loss']:.4f}, "
            f"val supported macro-F1@{args.threshold:.2f}={validation_f1:.4f}"
        )

        if validation_f1 > best_f1 + args.early_stopping_min_delta:
            best_f1 = validation_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model": model.state_dict(),
                "task": task,
                "classes": classes,
                "problem_type": "multi_label",
                "activation": "sigmoid",
                "threshold": args.threshold,
                "epoch": epoch,
                "trainable_backbone_stages": trainable_backbone_stages,
                "validation_metrics": validation_metrics,
            }
            temporary = output / "best.pt.part"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best.pt")
            write_json_atomic(output / "best_metrics.json", validation_metrics)
        else:
            epochs_without_improvement += 1

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; best supported macro-F1 "
                f"was {best_f1:.4f} at epoch {best_epoch}."
            )
            break

    print(
        f"Best validation supported macro-F1={best_f1:.4f} at epoch {best_epoch}; "
        f"checkpoint={output / 'best.pt'}"
    )
