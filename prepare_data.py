"""Safely unpack Kvasir-Capsule labelled images from the downloaded ZIP."""

from __future__ import annotations

import argparse
import io
import shutil
import tarfile
import zipfile
from pathlib import Path


def safe_name(name: str) -> str:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as outer:
        archives = [n for n in outer.namelist() if n.endswith(".tar.gz")]
        if not archives:
            raise RuntimeError("No .tar.gz class archives found in ZIP")
        for i, archive_name in enumerate(archives, 1):
            class_name = Path(archive_name).name.removesuffix(".tar.gz")
            class_dir = args.output / class_name
            class_dir.mkdir(exist_ok=True)
            print(f"[{i}/{len(archives)}] {class_name}")
            with outer.open(archive_name) as src:
                payload = io.BytesIO(src.read())
            with tarfile.open(fileobj=payload, mode="r:gz") as inner:
                for member in inner.getmembers():
                    if not member.isfile():
                        continue
                    safe_name(member.name)
                    suffix = Path(member.name).suffix.lower()
                    if suffix not in {".jpg", ".jpeg", ".png"}:
                        continue
                    extracted = inner.extractfile(member)
                    if extracted is None:
                        continue
                    target = class_dir / Path(member.name).name
                    with target.open("wb") as dst:
                        shutil.copyfileobj(extracted, dst)

    count = sum(1 for p in args.output.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"Done: {count} images in {args.output}")


if __name__ == "__main__":
    main()

