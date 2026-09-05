"""Train the isolated Galar Anatomy model."""

from pathlib import Path

from train_common import run_training


if __name__ == "__main__":
    run_training("anatomy", Path(__file__).resolve().parent / "runs" / "anatomy")
