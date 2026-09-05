"""Memory-efficient dataset for generated Galar multi-label metadata."""

from __future__ import annotations

import csv
import io
import json
import mmap
import struct
from array import array
from pathlib import Path
from typing import BinaryIO, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


EXPECTED_FIELDS = ("image_path", "study_id", "video_id", "frame_number", "labels")
CACHE_FORMAT = "galar_jpeg_shards_v1"
CACHE_INDEX_RECORD = struct.Struct("<QI")


class GalarMultiLabelDataset(Dataset):
    """Seek directly to CSV rows instead of holding millions of rows in RAM."""

    def __init__(
        self,
        metadata_path: Path,
        image_root: Path | None,
        classes: list[str],
        transform: Callable | None = None,
        max_samples: int | None = None,
        image_cache_root: Path | None = None,
    ) -> None:
        self.metadata_path = Path(metadata_path).resolve()
        self.image_root = Path(image_root).resolve() if image_root is not None else None
        self.image_cache_root = (
            Path(image_cache_root).resolve() if image_cache_root is not None else None
        )
        self.classes = list(classes)
        self.class_to_index = {label: index for index, label in enumerate(self.classes)}
        self.transform = transform
        self._offsets = array("Q")
        self._handle: BinaryIO | None = None
        self._cache_studies: dict[str, dict[str, object]] = {}
        self._cache_shard_handles: dict[str, BinaryIO] = {}
        self._cache_index_handles: dict[str, BinaryIO] = {}
        self._cache_index_maps: dict[str, mmap.mmap] = {}

        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Generated metadata not found: {self.metadata_path}")
        if self.image_cache_root is None:
            if self.image_root is None or not self.image_root.is_dir():
                raise FileNotFoundError(f"Galar image root not found: {self.image_root}")
        else:
            manifest_path = self.image_cache_root / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Complete Galar image cache manifest not found: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") != CACHE_FORMAT or manifest.get("status") != "complete":
                raise ValueError(f"Unsupported or incomplete Galar image cache: {manifest_path}")
            studies = manifest.get("studies")
            if not isinstance(studies, dict) or not studies:
                raise ValueError(f"Image cache has no study index: {manifest_path}")
            self._cache_studies = studies
        if not self.classes or len(self.classes) != len(self.class_to_index):
            raise ValueError("Class list must be non-empty and contain no duplicates")
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive when provided")

        with self.metadata_path.open("rb") as handle:
            header = handle.readline().decode("utf-8-sig").rstrip("\r\n")
            fields = tuple(next(csv.reader([header])))
            if fields != EXPECTED_FIELDS:
                raise ValueError(
                    f"Unexpected metadata columns in {self.metadata_path}: {fields}; "
                    f"expected {EXPECTED_FIELDS}"
                )
            while max_samples is None or len(self._offsets) < max_samples:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)
        if not self._offsets:
            raise ValueError(f"Generated metadata has no data rows: {self.metadata_path}")

    def __len__(self) -> int:
        return len(self._offsets)

    def _open(self) -> BinaryIO:
        if self._handle is None or self._handle.closed:
            self._handle = self.metadata_path.open("rb")
        return self._handle

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        handle = self._open()
        handle.seek(self._offsets[index])
        values = next(csv.reader([handle.readline().decode("utf-8")]))
        if len(values) != len(EXPECTED_FIELDS):
            raise ValueError(f"Malformed row {index + 2} in {self.metadata_path}")
        row = dict(zip(EXPECTED_FIELDS, values, strict=True))

        target = torch.zeros(len(self.classes), dtype=torch.float32)
        labels = [label for label in row["labels"].split("|") if label]
        unknown = sorted(set(labels) - set(self.class_to_index))
        if unknown:
            raise ValueError(
                f"Unknown labels {unknown} at row {index + 2} in {self.metadata_path}"
            )
        for label in labels:
            target[self.class_to_index[label]] = 1.0

        image_description: Path | str
        try:
            if self.image_cache_root is not None:
                encoded = self._read_cached_frame(row["study_id"], int(row["frame_number"]))
                image_description = (
                    f"cache={self.image_cache_root}, study={row['study_id']}, "
                    f"frame={row['frame_number']}"
                )
                with Image.open(io.BytesIO(encoded)) as opened:
                    image = opened.convert("RGB")
            else:
                assert self.image_root is not None
                image_path = self.image_root / Path(row["image_path"])
                image_description = image_path
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        except Exception as error:
            raise RuntimeError(
                f"Could not load Galar image for metadata row {index + 2}: "
                f"{locals().get('image_description', row['image_path'])}"
            ) from error
        return image, target

    def _read_cached_frame(self, study_id: str, frame_number: int) -> bytes:
        study = self._cache_studies.get(study_id)
        if not isinstance(study, dict):
            raise KeyError(f"Study {study_id} is absent from the SSD image cache")
        if frame_number < 0 or frame_number > int(study["max_frame_number"]):
            raise KeyError(
                f"Frame {frame_number} is outside the cache index for study {study_id}"
            )
        if study_id not in self._cache_index_maps:
            assert self.image_cache_root is not None
            shard_path = (self.image_cache_root / str(study["shard"])).resolve()
            index_path = (self.image_cache_root / str(study["index"])).resolve()
            if (
                not shard_path.is_relative_to(self.image_cache_root)
                or not index_path.is_relative_to(self.image_cache_root)
            ):
                raise ValueError(f"Unsafe cache paths for study {study_id}")
            shard_handle = shard_path.open("rb")
            index_handle = index_path.open("rb")
            expected_index_bytes = (int(study["max_frame_number"]) + 1) * CACHE_INDEX_RECORD.size
            if (
                index_path.stat().st_size != expected_index_bytes
                or shard_path.stat().st_size != int(study["shard_bytes"])
            ):
                shard_handle.close()
                index_handle.close()
                raise ValueError(f"Corrupt cache file size for study {study_id}")
            self._cache_shard_handles[study_id] = shard_handle
            self._cache_index_handles[study_id] = index_handle
            self._cache_index_maps[study_id] = mmap.mmap(
                index_handle.fileno(), length=0, access=mmap.ACCESS_READ
            )
        index_map = self._cache_index_maps[study_id]
        offset, length = CACHE_INDEX_RECORD.unpack_from(
            index_map, frame_number * CACHE_INDEX_RECORD.size
        )
        if length == 0:
            raise KeyError(f"Frame {frame_number} of study {study_id} is absent from the cache")
        shard_handle = self._cache_shard_handles[study_id]
        shard_handle.seek(offset)
        encoded = shard_handle.read(length)
        if len(encoded) != length:
            raise EOFError(f"Short cache read for study {study_id}, frame {frame_number}")
        return encoded

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_cache_shard_handles"] = {}
        state["_cache_index_handles"] = {}
        state["_cache_index_maps"] = {}
        return state

    def close(self) -> None:
        """Close metadata, shard, and memory-mapped index handles."""

        handle = getattr(self, "_handle", None)
        if handle is not None and not handle.closed:
            handle.close()
        for index_map in getattr(self, "_cache_index_maps", {}).values():
            index_map.close()
        for cache_handle in getattr(self, "_cache_index_handles", {}).values():
            if not cache_handle.closed:
                cache_handle.close()
        for cache_handle in getattr(self, "_cache_shard_handles", {}).values():
            if not cache_handle.closed:
                cache_handle.close()
        getattr(self, "_cache_index_maps", {}).clear()
        getattr(self, "_cache_index_handles", {}).clear()
        getattr(self, "_cache_shard_handles", {}).clear()

    def __enter__(self) -> GalarMultiLabelDataset:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def metadata_study_ids(path: Path) -> set[str]:
    """Read distinct source studies from one generated metadata file."""

    studies: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "study_id" not in reader.fieldnames:
            raise ValueError(f"Metadata is missing study_id: {path}")
        for row in reader:
            studies.add(row["study_id"])
    return studies
