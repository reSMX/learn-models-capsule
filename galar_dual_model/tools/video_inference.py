"""Frame-level inference for a video with the two Galar models.

The generated overlay is a research aid. It does not add temporal reasoning and
must not be interpreted as a medical diagnosis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import v2
from tqdm import tqdm

from galar_dual_model.labels import ANATOMY_LABELS, PATHOLOGY_LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANATOMY_MODEL = (
    PROJECT_ROOT / "galar_dual_model/runs/anatomy_workers0_20260907/best.pt"
)
DEFAULT_PATHOLOGY_MODEL = PROJECT_ROOT / "galar_dual_model/runs/pathology/best.pt"
IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Покадрово обработать видео двумя моделями Galar и сохранить копию "
            "с оценками модели."
        )
    )
    parser.add_argument("--input", type=Path, help="Путь к исходному видео")
    parser.add_argument("--output", type=Path, help="Путь к выходному MP4")
    parser.add_argument("--anatomy-model", type=Path, default=DEFAULT_ANATOMY_MODEL)
    parser.add_argument("--pathology-model", type=Path, default=DEFAULT_PATHOLOGY_MODEL)
    parser.add_argument(
        "--anatomy-threshold",
        type=float,
        help="Переопределить порог Anatomy из контрольной точки",
    )
    parser.add_argument(
        "--pathology-threshold",
        type=float,
        help="Переопределить порог Pathology из контрольной точки",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cpu", action="store_true", help="Не использовать CUDA")
    return parser.parse_args()


def ask_video_path() -> Path:
    raw = input("Путь к видео: ").strip().strip('"').strip("'")
    return Path(raw).expanduser()


def _expected_classes(task: str) -> list[str]:
    if task == "anatomy":
        return list(ANATOMY_LABELS)
    if task == "pathology":
        return list(PATHOLOGY_LABELS)
    raise ValueError(f"Неизвестная задача: {task}")


def load_model(
    path: Path,
    expected_task: str,
    device: torch.device,
) -> tuple[nn.Module, list[str], float, dict[str, Any]]:
    """Load and validate one checkpoint produced by train_common.py."""

    if not path.is_file():
        raise FileNotFoundError(f"Контрольная точка не найдена: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Неподдерживаемая структура контрольной точки: {path}")
    if checkpoint.get("task") != expected_task:
        raise RuntimeError(
            f"Ожидалась задача {expected_task!r}, получена {checkpoint.get('task')!r}: {path}"
        )
    if checkpoint.get("problem_type") != "multi_label":
        raise RuntimeError(f"Контрольная точка не является многометочной: {path}")
    if checkpoint.get("activation") != "sigmoid":
        raise RuntimeError(f"Контрольная точка не использует sigmoid: {path}")

    classes = checkpoint.get("classes")
    expected_classes = _expected_classes(expected_task)
    if not isinstance(classes, list) or classes != expected_classes:
        raise RuntimeError(
            f"Порядок меток {expected_task} не совпадает с galar_dual_model/labels.py"
        )

    threshold = checkpoint.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0.0 < float(threshold) < 1.0:
        raise RuntimeError(f"В контрольной точке нет корректного порога: {path}")

    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"В контрольной точке нет весов модели: {path}")

    model = efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, len(classes))
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Веса несовместимы с EfficientNet-B0 для задачи {expected_task}: {path}"
        ) from error

    model.eval().to(device)
    return model, list(classes), float(threshold), checkpoint


def make_transform() -> v2.Compose:
    weights = EfficientNet_B0_Weights.DEFAULT
    transform = weights.transforms()
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(IMAGE_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=transform.mean, std=transform.std),
        ]
    )


def frame_to_tensor(frame: np.ndarray, transform: v2.Compose) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return transform(Image.fromarray(rgb))


def predict_scores(
    model: nn.Module, batch: torch.Tensor, device: torch.device
) -> np.ndarray:
    batch = batch.to(device, non_blocking=device.type == "cuda")
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch)
        else:
            logits = model(batch)
        scores = torch.sigmoid(logits.float())
    return scores.cpu().numpy()


def wrap_text(
    text: str, max_width: int, font_scale: float, thickness: int
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        width = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0][0]
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_panel(
    frame: np.ndarray,
    lines: Sequence[str],
    top: bool,
    font_scale: float,
    thickness: int,
) -> None:
    height, width = frame.shape[:2]
    line_height = max(22, int(30 * font_scale))
    padding = 10
    panel_height = padding * 2 + line_height * len(lines)
    y1 = 8 if top else height - panel_height - 8
    y2 = y1 + panel_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, y1), (width - 9, y2), BLACK, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for index, line in enumerate(lines):
        baseline = y1 + padding + line_height * (index + 1) - 6
        cv2.putText(
            frame,
            line,
            (18, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            WHITE,
            thickness,
            cv2.LINE_AA,
        )


def _result_text(
    title: str,
    labels: Sequence[str],
    scores: np.ndarray,
    threshold: float,
) -> str:
    detected = sorted(
        (
            (label, float(scores[index]))
            for index, label in enumerate(labels)
            if float(scores[index]) >= threshold
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if detected:
        values = ", ".join(f"{label} [{score:.2f}]" for label, score in detected)
        return f"{title}: {values}"

    top_index = int(np.argmax(scores))
    return (
        f"{title}: no score reached {threshold:.2f}; "
        f"top {labels[top_index]} [{float(scores[top_index]):.2f}]"
    )


def annotate_frame(
    frame: np.ndarray,
    anatomy_scores: np.ndarray,
    pathology_scores: np.ndarray,
    anatomy_labels: Sequence[str],
    pathology_labels: Sequence[str],
    anatomy_threshold: float,
    pathology_threshold: float,
) -> np.ndarray:
    width = frame.shape[1]
    font_scale = min(0.8, max(0.42, width / 1200.0))
    thickness = 2 if width >= 700 else 1
    max_text_width = max(100, width - 44)

    pathology_lines = wrap_text(
        _result_text(
            "Pathology model scores",
            pathology_labels,
            pathology_scores,
            pathology_threshold,
        ),
        max_text_width,
        font_scale,
        thickness,
    )
    anatomy_lines = wrap_text(
        _result_text(
            "Anatomy model scores",
            anatomy_labels,
            anatomy_scores,
            anatomy_threshold,
        ),
        max_text_width,
        font_scale,
        thickness,
    )
    draw_panel(frame, pathology_lines, top=True, font_scale=font_scale, thickness=thickness)
    draw_panel(frame, anatomy_lines, top=False, font_scale=font_scale, thickness=thickness)
    return frame


def open_video(
    input_path: Path, output_path: Path
) -> tuple[cv2.VideoCapture, cv2.VideoWriter, int]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть исходное видео: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Исходное видео имеет некорректный размер кадра")
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Не удалось создать выходное видео: {output_path}")
    return capture, writer, total


def process_video(
    input_path: Path,
    output_path: Path,
    anatomy_model: nn.Module,
    pathology_model: nn.Module,
    anatomy_labels: Sequence[str],
    pathology_labels: Sequence[str],
    device: torch.device,
    anatomy_threshold: float,
    pathology_threshold: float,
    batch_size: int,
) -> None:
    capture, writer, total = open_video(input_path, output_path)
    transform = make_transform()
    try:
        with tqdm(
            total=total if total > 0 else None,
            unit="кадр",
            desc="Обработка",
        ) as progress:
            finished = False
            while not finished:
                frames: list[np.ndarray] = []
                tensors: list[torch.Tensor] = []
                for _ in range(batch_size):
                    ok, frame = capture.read()
                    if not ok:
                        finished = True
                        break
                    frames.append(frame)
                    tensors.append(frame_to_tensor(frame, transform))

                if not frames:
                    break

                batch = torch.stack(tensors)
                anatomy_scores = predict_scores(anatomy_model, batch, device)
                pathology_scores = predict_scores(pathology_model, batch, device)
                for frame, anatomy_row, pathology_row in zip(
                    frames, anatomy_scores, pathology_scores
                ):
                    writer.write(
                        annotate_frame(
                            frame,
                            anatomy_row,
                            pathology_row,
                            anatomy_labels,
                            pathology_labels,
                            anatomy_threshold,
                            pathology_threshold,
                        )
                    )
                progress.update(len(frames))
    finally:
        capture.release()
        writer.release()


def _validate_threshold(value: float | None, name: str) -> None:
    if value is not None and not 0.0 < value < 1.0:
        raise ValueError(f"{name} должен находиться между 0 и 1")


def main() -> None:
    args = parse_args()
    input_path = (args.input or ask_video_path()).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Видео не найдено: {input_path}")
    _validate_threshold(args.anatomy_threshold, "--anatomy-threshold")
    _validate_threshold(args.pathology_threshold, "--pathology-threshold")
    if args.batch_size < 1:
        raise ValueError("--batch-size должен быть не меньше 1")

    output_path = (
        args.output or input_path.with_name(f"{input_path.stem}_annotated.mp4")
    ).resolve()
    if output_path == input_path:
        raise ValueError("Пути исходного и выходного видео должны различаться")
    if output_path.exists():
        raise FileExistsError(f"Выходной файл уже существует: {output_path}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Устройство: {device}")
    anatomy_model, anatomy_labels, anatomy_default, _ = load_model(
        args.anatomy_model.resolve(), "anatomy", device
    )
    pathology_model, pathology_labels, pathology_default, _ = load_model(
        args.pathology_model.resolve(), "pathology", device
    )
    anatomy_threshold = args.anatomy_threshold or anatomy_default
    pathology_threshold = args.pathology_threshold or pathology_default

    process_video(
        input_path=input_path,
        output_path=output_path,
        anatomy_model=anatomy_model,
        pathology_model=pathology_model,
        anatomy_labels=anatomy_labels,
        pathology_labels=pathology_labels,
        device=device,
        anatomy_threshold=anatomy_threshold,
        pathology_threshold=pathology_threshold,
        batch_size=args.batch_size,
    )
    print(f"Сохранено: {output_path}")
    print("Исследовательский результат; оценки моделей не являются диагнозом.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1)
