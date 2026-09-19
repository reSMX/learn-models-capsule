"""Save the strongest Pathology frames and a complete multi-label report."""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from galar_dual_model.tools.video_inference import (
    DEFAULT_PATHOLOGY_MODEL,
    frame_to_tensor,
    load_model,
    make_transform,
    predict_scores,
)


DEFAULT_COUNT = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_MIN_GAP_SECONDS = 2.0


@dataclass(order=True)
class Candidate:
    score: float
    frame_index: int
    top_class_index: int
    scores: tuple[float, ...] = field(compare=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Сохранить кадры видео с наиболее высокими оценками Pathology "
            "без нанесения подписей на изображения."
        )
    )
    parser.add_argument("--input", type=Path, help="Путь к исходному видео")
    parser.add_argument("--output-dir", type=Path, help="Каталог результата")
    parser.add_argument("--model", type=Path, default=DEFAULT_PATHOLOGY_MODEL)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument(
        "--threshold",
        type=float,
        help="Переопределить порог из контрольной точки",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-gap-seconds", type=float, default=DEFAULT_MIN_GAP_SECONDS)
    parser.add_argument("--cpu", action="store_true", help="Не использовать CUDA")
    return parser.parse_args()


def ask_video_path() -> Path:
    raw = input("Путь к видео: ").strip().strip('"').strip("'")
    return Path(raw).expanduser()


def unique_directory(path: Path) -> Path:
    if not path.exists():
        return path
    number = 2
    while True:
        candidate = path.with_name(f"{path.name}_{number}")
        if not candidate.exists():
            return candidate
        number += 1


def scan_video(
    video_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    batch_size: int,
    pool_size: int,
) -> tuple[list[Candidate], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    transform = make_transform()
    candidates: list[Candidate] = []
    next_frame_index = 0

    try:
        with tqdm(
            total=total if total > 0 else None,
            unit="кадр",
            desc="Поиск",
        ) as progress:
            finished = False
            while not finished:
                tensors: list[torch.Tensor] = []
                indices: list[int] = []
                for _ in range(batch_size):
                    ok, frame = capture.read()
                    if not ok:
                        finished = True
                        break
                    indices.append(next_frame_index)
                    next_frame_index += 1
                    tensors.append(frame_to_tensor(frame, transform))

                if not tensors:
                    break

                scores = predict_scores(model, torch.stack(tensors), device)
                for frame_index, row in zip(indices, scores):
                    class_index = int(np.argmax(row))
                    score = float(row[class_index])
                    if score < threshold:
                        continue
                    item = Candidate(
                        score=score,
                        frame_index=frame_index,
                        top_class_index=class_index,
                        scores=tuple(float(value) for value in row),
                    )
                    if len(candidates) < pool_size:
                        heapq.heappush(candidates, item)
                    elif item > candidates[0]:
                        heapq.heapreplace(candidates, item)
                progress.update(len(tensors))
    finally:
        capture.release()

    return sorted(candidates, reverse=True), fps


def select_frames(
    candidates: list[Candidate],
    count: int,
    fps: float,
    min_gap_seconds: float,
) -> list[Candidate]:
    gap = max(0, round(fps * min_gap_seconds))
    selected: list[Candidate] = []
    selected_indices: set[int] = set()

    for item in candidates:
        if all(abs(item.frame_index - other.frame_index) >= gap for other in selected):
            selected.append(item)
            selected_indices.add(item.frame_index)
            if len(selected) == count:
                return selected

    for item in candidates:
        if item.frame_index in selected_indices:
            continue
        selected.append(item)
        selected_indices.add(item.frame_index)
        if len(selected) == count:
            break
    return selected


def safe_name(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    return result.strip("_") or "pathology"


def write_png(path: Path, frame: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(f"Не удалось закодировать кадр: {path}")
    encoded.tofile(str(path))


def save_frames(
    video_path: Path,
    output_dir: Path,
    selected: list[Candidate],
    labels: list[str],
    threshold: float,
    fps: float,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось повторно открыть видео: {video_path}")

    report: list[dict[str, object]] = []
    try:
        for rank, item in enumerate(selected, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, item.frame_index)
            ok, frame = capture.read()
            if not ok:
                print(
                    f"Не удалось прочитать кадр {item.frame_index}; кадр пропущен.",
                    file=sys.stderr,
                )
                continue

            seconds = item.frame_index / fps
            label = safe_name(labels[item.top_class_index])
            filename = (
                f"{rank:02d}_frame_{item.frame_index:08d}_"
                f"time_{seconds:.2f}s_{label}_score_{item.score:.3f}.png"
            )
            write_png(output_dir / filename, frame)
            detected = [
                {"class_key": labels[index], "score": score}
                for index, score in enumerate(item.scores)
                if score >= threshold
            ]
            detected.sort(key=lambda value: float(value["score"]), reverse=True)
            report.append(
                {
                    "file": filename,
                    "frame_index": item.frame_index,
                    "time_seconds": seconds,
                    "top_class": labels[item.top_class_index],
                    "top_score": item.score,
                    "detected": detected,
                    "all_scores": dict(zip(labels, item.scores)),
                }
            )
    finally:
        capture.release()
    return report


def main() -> None:
    args = parse_args()
    video_path = (args.input or ask_video_path()).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")
    if args.count < 1:
        raise ValueError("--count должен быть не меньше 1")
    if args.threshold is not None and not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold должен находиться между 0 и 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size должен быть не меньше 1")
    if args.min_gap_seconds < 0:
        raise ValueError("--min-gap-seconds не может быть отрицательным")

    output_dir = args.output_dir or video_path.with_name(
        f"{video_path.stem}_pathology_frames"
    )
    output_dir = unique_directory(output_dir.resolve())
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Устройство: {device}")
    model, labels, checkpoint_threshold, checkpoint = load_model(
        args.model.resolve(), "pathology", device
    )
    threshold = args.threshold or checkpoint_threshold

    pool_size = max(args.count * 500, args.count)
    candidates, fps = scan_video(
        video_path=video_path,
        model=model,
        device=device,
        threshold=threshold,
        batch_size=args.batch_size,
        pool_size=pool_size,
    )
    if not candidates:
        print(f"Ни одна оценка Pathology не достигла порога {threshold:.2f}.")
        return

    selected = select_frames(
        candidates=candidates,
        count=args.count,
        fps=fps,
        min_gap_seconds=args.min_gap_seconds,
    )
    frames = save_frames(video_path, output_dir, selected, labels, threshold, fps)
    summary = {
        "source_video": video_path.name,
        "model_checkpoint": args.model.name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": threshold,
        "requested_count": args.count,
        "saved_count": len(frames),
        "frames": frames,
        "warning": (
            "Исследовательский результат. Оценки модели не являются диагнозом; "
            "отсутствие оценки выше порога не исключает патологию."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Сохранено кадров: {len(frames)}. Каталог: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1)
