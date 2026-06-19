"""VisDrone dataset preparation utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

VALID_CATEGORY_MIN = 1
VALID_CATEGORY_MAX = len(VISDRONE_NAMES)


@dataclass(frozen=True)
class VisDroneSplit:
    name: str
    root: Path
    images: Path
    annotations: Path
    yolo_labels: bool = field(default=False)  # True when labels/ is already YOLO format


@dataclass(frozen=True)
class PreparedDataset:
    root: Path
    yaml_path: Path
    split: str
    images: Path
    labels: Path
    image_count: int
    label_count: int
    skipped_box_count: int


def find_visdrone_splits(data_root: Path) -> dict[str, VisDroneSplit]:
    """Find VisDrone DET split directories under a Kaggle dataset mount.

    Supports both the original VisDrone layout (annotations/) and
    pre-converted Kaggle datasets that store YOLO labels in labels/.
    """
    data_root = data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data root does not exist: {data_root}")

    candidates: list[VisDroneSplit] = []

    # Prefer original VisDrone format: split_root/annotations/ + split_root/images/
    for annotation_dir in data_root.rglob("annotations"):
        split_root = annotation_dir.parent
        images_dir = split_root / "images"
        if images_dir.is_dir() and annotation_dir.is_dir():
            name = _infer_split_name(split_root)
            candidates.append(VisDroneSplit(name=name, root=split_root, images=images_dir, annotations=annotation_dir, yolo_labels=False))

    # Fall back to pre-converted YOLO format: split_root/labels/ + split_root/images/
    if not candidates:
        for labels_dir in data_root.rglob("labels"):
            split_root = labels_dir.parent
            images_dir = split_root / "images"
            if images_dir.is_dir() and labels_dir.is_dir():
                name = _infer_split_name(split_root)
                candidates.append(VisDroneSplit(name=name, root=split_root, images=images_dir, annotations=labels_dir, yolo_labels=True))

    if not candidates:
        raise FileNotFoundError(f"could not find VisDrone images/annotations split under {data_root}")

    splits: dict[str, VisDroneSplit] = {}
    for split in sorted(candidates, key=lambda item: len(str(item.root))):
        splits.setdefault(split.name, split)
    return splits


def prepare_yolo_dataset(data_root: Path, output_root: Path, split: str = "val") -> PreparedDataset:
    """Convert one VisDrone DET split to YOLO labels for Ultralytics validation."""
    splits = find_visdrone_splits(data_root)
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise KeyError(f"split {split!r} not found under {data_root}; available: {available}")

    source = splits[split]
    output_root = output_root.expanduser().resolve()
    image_link = output_root / "images" / split
    image_link.parent.mkdir(parents=True, exist_ok=True)
    _ensure_image_link(source.images, image_link)

    image_paths = sorted(path for path in source.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    if source.yolo_labels:
        # Labels are already in YOLO format — symlink the directory directly.
        label_dir = output_root / "labels" / split
        label_dir.parent.mkdir(parents=True, exist_ok=True)
        _ensure_image_link(source.annotations, label_dir)
        label_count = sum(1 for p in source.annotations.iterdir() if p.suffix == ".txt")
        skipped_count = 0
    else:
        # Convert from original VisDrone CSV annotation format.
        label_dir = output_root / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        label_count = 0
        skipped_count = 0
        for image_path in image_paths:
            annotation_path = source.annotations / f"{image_path.stem}.txt"
            output_label = label_dir / f"{image_path.stem}.txt"
            labels, skipped = convert_annotation_file(annotation_path, image_path)
            skipped_count += skipped
            label_count += len(labels)
            output_label.write_text("".join(labels), encoding="utf-8")

    yaml_path = output_root / "visdrone.yaml"
    yaml_payload = {
        "path": str(output_root),
        "train": f"images/{split}",
        "val": f"images/{split}",
        "test": f"images/{split}",
        "nc": len(VISDRONE_NAMES),
        "names": {idx: name for idx, name in enumerate(VISDRONE_NAMES)},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")

    return PreparedDataset(
        root=output_root,
        yaml_path=yaml_path,
        split=split,
        images=image_link,
        labels=label_dir,
        image_count=len(image_paths),
        label_count=label_count,
        skipped_box_count=skipped_count,
    )


def convert_annotation_file(annotation_path: Path, image_path: Path) -> tuple[list[str], int]:
    """Convert one VisDrone annotation file to YOLO label rows."""
    width, height = _read_image_size(image_path)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size for {image_path}: {width}x{height}")

    labels: list[str] = []
    skipped = 0
    if not annotation_path.exists():
        return labels, skipped

    for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split(",")
        if len(parts) < 6:
            skipped += 1
            continue
        left, top, box_width, box_height = (float(parts[idx]) for idx in range(4))
        category = int(float(parts[5]))
        if category < VALID_CATEGORY_MIN or category > VALID_CATEGORY_MAX or box_width <= 0 or box_height <= 0:
            skipped += 1
            continue
        x_center = (left + box_width / 2.0) / width
        y_center = (top + box_height / 2.0) / height
        norm_width = box_width / width
        norm_height = box_height / height
        if not _is_normalized_box_valid(x_center, y_center, norm_width, norm_height):
            skipped += 1
            continue
        class_id = category - 1
        labels.append(f"{class_id} {x_center:.8f} {y_center:.8f} {norm_width:.8f} {norm_height:.8f}\n")
    return labels, skipped


def _infer_split_name(path: Path) -> str:
    lower = str(path).lower()
    if "val" in lower:
        return "val"
    if "train" in lower:
        return "train"
    if "test" in lower:
        return "test"
    return path.name.lower().replace("visdrone2019-det-", "")


def _ensure_image_link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.resolve() == source.resolve():
            return
        raise FileExistsError(f"image target already exists and points elsewhere: {target}")
    os.symlink(source, target, target_is_directory=True)


def _is_normalized_box_valid(x_center: float, y_center: float, width: float, height: float) -> bool:
    return width > 0 and height > 0 and 0 <= x_center <= 1 and 0 <= y_center <= 1 and width <= 1 and height <= 1


def _read_image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Kaggle/Ultralytics provides pillow.
        raise RuntimeError("Pillow is required to read image sizes") from exc
    with Image.open(path) as image:
        return image.size
