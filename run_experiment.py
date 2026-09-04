from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Keep the user-facing junction path on Windows. Path.resolve() follows this workspace's
# junction to a Cyrillic directory name, which some PowerShell encodings display incorrectly.
PROJECT_ROOT = Path(__file__).absolute().parent
IMAGES = Path(r"D:\dataset_quazir")
METADATA = Path(r"D:\dataset_quazir\metadata_with_galar.csv")

EXPERIMENTS = {
    "head-only": {
        "output": Path("runs/combined_galar_head_only"),
        "arguments": ["--backbone-strategy", "head-only"],
    },
    "last-block": {
        "output": Path("runs/combined_galar_last_block"),
        "arguments": [
            "--backbone-strategy",
            "staged",
            "--freeze-backbone-epochs",
            "2",
            "--trainable-backbone-blocks",
            "1",
            "--backbone-lr-multiplier",
            "0.03",
        ],
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch a reproducible combined-data backbone ablation experiment."
    )
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the experiment output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit data and splits without loading or training the model",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved command without running it",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers cannot be negative")
    if not IMAGES.is_dir():
        parser.error(f"image root does not exist: {IMAGES}")
    if not METADATA.is_file():
        parser.error(f"combined metadata does not exist: {METADATA}")

    preset = EXPERIMENTS[args.experiment]
    output = args.output or preset["output"]
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    completed_artifacts = [output / "best.pt", output / "history.csv"]
    existing_completed = [path for path in completed_artifacts if path.exists()]
    if existing_completed and not args.dry_run:
        parser.error(
            "refusing to overwrite an existing training run; choose --output with a new "
            f"directory (found: {', '.join(str(path) for path in existing_completed)})"
        )

    command = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "--images",
        str(IMAGES),
        "--metadata",
        str(METADATA),
        "--output",
        str(output),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        *preset["arguments"],
    ]
    if args.dry_run:
        command.append("--dry-run")

    print(f"Experiment: {args.experiment}")
    print(f"Output: {output}")
    print(subprocess.list2cmdline(command))
    if args.print_command:
        return 0

    try:
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
