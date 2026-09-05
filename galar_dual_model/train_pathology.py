"""Train the isolated Galar Pathology model."""

from pathlib import Path

from train_common import run_training


if __name__ == "__main__":
    run_training("pathology", Path(__file__).resolve().parent / "runs" / "pathology")
