"""YOLOv26x inference visualization — academic GT + prediction overlay."""

from __future__ import annotations

import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from .visdrone import prepare_yolo_dataset, VISDRONE_NAMES

# BGR color palette (OpenCV convention: Blue, Green, Red)
_GT_COLOR: tuple[int, int, int] = (94, 197, 34)    # vivid green — ground truth
_PRED_COLOR: tuple[int, int, int] = (68, 68, 239)  # vivid red — predictions
_INFO_BAR_H = 34                                    # pixel height of bottom info bar


def run_yolov26x_inference(
    data_root: Path,
    work_dir: Path,
    split: str = "val",
    model_name: str = "yolo26x.pt",
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "0,1",
    max_frames: int = 300,
    fps: float = 5.0,
    wandb_project: str = "distillNas",
    wandb_entity: str | None = None,
    run_name: str | None = None,
) -> Path:
    """Run YOLOv26x predict on VisDrone val, write GT+pred annotated video, upload to W&B."""
    import cv2
    import numpy as np
    from ultralytics import YOLO
    import wandb

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    work_dir = work_dir.expanduser().resolve()
    dataset_dir = work_dir / "visdrone_yolo"
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_yolo_dataset(data_root=data_root, output_root=dataset_dir, split=split)
    git_sha = _git_sha(Path.cwd())
    run_name = run_name or "yolov26x-infer"

    image_paths = sorted(
        p for p in prepared.images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if max_frames and max_frames < len(image_paths):
        image_paths = image_paths[:max_frames]

    frame_count = len(image_paths)
    print(f"Running inference on {frame_count} frames from {prepared.images}", flush=True)

    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        job_type="inference-visualize",
        tags=["visdrone", "yolov26x", "inference", "visualization", "kaggle", "baseline"],
        config={
            "model": model_name,
            "baseline_model": "YOLOv26x",
            "dataset": "VisDrone",
            "kaggle_dataset_slug": "banuprasadb/visdrone-dataset",
            "split": split,
            "imgsz": imgsz,
            "conf": conf,
            "device": device,
            "max_frames": max_frames,
            "frame_count": frame_count,
            "fps": fps,
            "visualization": "gt_and_pred",
            "accelerator": "NvidiaTeslaT4",
            "git_sha": git_sha,
            "python": platform.python_version(),
        },
    )

    video_path = results_dir / "yolov26x_visdrone_inference.mp4"
    sample_step = max(1, frame_count // 5)
    sample_indices = {i * sample_step for i in range(5)}
    samples: list[tuple[Any, str]] = []  # (RGB ndarray, caption)

    try:
        model = YOLO(model_name)

        probe = cv2.imread(str(image_paths[0]))
        if probe is None:
            raise RuntimeError(f"Could not read image: {image_paths[0]}")
        img_h, img_w = probe.shape[:2]

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (img_w, img_h + _INFO_BAR_H),
        )

        written = 0
        for result in model.predict(
            source=[str(p) for p in image_paths],
            imgsz=imgsz,
            conf=conf,
            device=device,
            half=True,
            stream=True,
            verbose=False,
        ):
            raw = result.orig_img  # BGR, native resolution
            label_path = prepared.labels / f"{Path(result.path).stem}.txt"

            frame, gt_n, pred_n = _compose_frame(
                img=raw,
                result=result,
                label_path=label_path,
                names=VISDRONE_NAMES,
                frame_idx=written,
                total_frames=frame_count,
                conf_thresh=conf,
                img_w=img_w,
            )
            writer.write(frame)

            if written in sample_indices:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cap = (
                    f"{Path(result.path).name}  |  "
                    f"GT: {gt_n} boxes  |  Pred: {pred_n} boxes"
                )
                samples.append((rgb, cap))

            written += 1
            if written % 50 == 0:
                print(f"  wrote {written}/{frame_count} frames", flush=True)

        writer.release()
        print(f"Video saved: {video_path} ({written} frames)", flush=True)

        wandb_run.log({
            "inference/video": wandb.Video(str(video_path), fps=fps, format="mp4"),
            "inference/frame_count": written,
            "inference/samples": [wandb.Image(rgb, caption=cap) for rgb, cap in samples],
        })
        wandb_run.summary.update({
            "inference/frame_count": written,
            "inference/video_path": str(video_path),
            "run/git_sha": git_sha,
        })

        art = wandb.Artifact(
            "yolov26x-visdrone-inference-video",
            type="visualization",
            metadata={
                "model": model_name,
                "split": split,
                "frame_count": written,
                "fps": fps,
                "conf": conf,
                "imgsz": imgsz,
                "visualization": "gt_and_pred",
                "git_sha": git_sha,
            },
        )
        art.add_file(str(video_path))
        wandb_run.log_artifact(art)

    finally:
        wandb.finish()

    return video_path


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------

def _compose_frame(
    img: Any,
    result: Any,
    label_path: Path,
    names: list[str],
    frame_idx: int,
    total_frames: int,
    conf_thresh: float,
    img_w: int,
) -> tuple[Any, int, int]:
    """Return (annotated_frame_with_info_bar, gt_count, pred_count)."""
    import cv2
    import numpy as np

    img_h = img.shape[0]
    frame = img.copy()

    # --- Ground-truth boxes (dashed green) ---
    gt_boxes = _load_gt_boxes(label_path, img_w, img_h)
    for x1, y1, x2, y2, cls_id in gt_boxes:
        _draw_dashed_rect(frame, (x1, y1), (x2, y2), _GT_COLOR)
        _draw_label(frame, names[cls_id], x1, y1, img_h, _GT_COLOR)

    # --- Prediction boxes (solid red) ---
    # Use result.names (the model's own class map) — pretrained COCO has 80 classes,
    # not 10, so VISDRONE_NAMES would index out of range for most COCO class IDs.
    pred_names = result.names  # dict {class_id: class_name}
    pred_boxes: list[tuple[int, int, int, int, int, float]] = []
    if result.boxes is not None and len(result.boxes):
        for box in result.boxes:
            bx1, by1, bx2, by2 = (int(v) for v in box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            pred_boxes.append((bx1, by1, bx2, by2, cls_id, conf))
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), _PRED_COLOR, thickness=2, lineType=cv2.LINE_AA)
            label = f"{pred_names.get(cls_id, str(cls_id))} {conf:.0%}"
            _draw_label(frame, label, bx1, by1, img_h, _PRED_COLOR)

    _draw_legend(frame, conf_thresh)

    info_bar = _make_info_bar(img_w, frame_idx, total_frames, len(gt_boxes), len(pred_boxes), conf_thresh)
    return np.vstack([frame, info_bar]), len(gt_boxes), len(pred_boxes)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, int, int, int, int]]:
    """Parse a YOLO label file and return pixel-space (x1,y1,x2,y2,class_id) tuples."""
    if not label_path.exists():
        return []
    boxes: list[tuple[int, int, int, int, int]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = max(0, int((cx - bw / 2) * img_w))
        y1 = max(0, int((cy - bh / 2) * img_h))
        x2 = min(img_w - 1, int((cx + bw / 2) * img_w))
        y2 = min(img_h - 1, int((cy + bh / 2) * img_h))
        boxes.append((x1, y1, x2, y2, cls_id))
    return boxes


def _draw_dashed_rect(
    img: Any,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 8,
    gap_len: int = 5,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    for start, end in [
        ((x1, y1), (x2, y1)),
        ((x1, y2), (x2, y2)),
        ((x1, y1), (x1, y2)),
        ((x2, y1), (x2, y2)),
    ]:
        _draw_dashed_segment(img, start, end, color, thickness, dash_len, gap_len)


def _draw_dashed_segment(
    img: Any,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_len: int,
    gap_len: int,
) -> None:
    import cv2

    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    draw = True
    traveled = 0.0
    while traveled < dist:
        seg = dash_len if draw else gap_len
        end_t = min(traveled + seg, dist)
        if draw:
            t0, t1 = traveled / dist, end_t / dist
            p0 = (round(x1 + t0 * dx), round(y1 + t0 * dy))
            p1 = (round(x1 + t1 * dx), round(y1 + t1 * dy))
            cv2.line(img, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        traveled += seg
        draw = not draw


def _draw_label(
    img: Any,
    text: str,
    x1: int,
    y1: int,
    img_h: int,
    color: tuple[int, int, int],
    font_scale: float = 0.42,
    thickness: int = 1,
) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3
    # Place label above the box; flip below if it would go off-frame
    if y1 - th - 2 * pad >= 0:
        bg_top, bg_bot = y1 - th - 2 * pad, y1
        text_y = y1 - pad - baseline // 2
    else:
        bg_top, bg_bot = y1, y1 + th + 2 * pad
        text_y = y1 + th + pad - baseline // 2
    cv2.rectangle(img, (x1, bg_top), (x1 + tw + 2 * pad, bg_bot), color, -1)
    cv2.putText(img, text, (x1 + pad, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_legend(img: Any, conf_thresh: float) -> None:
    import cv2

    h, w = img.shape[:2]
    panel_w, panel_h = 218, 66
    x0, y0 = w - panel_w - 10, 10

    # Semi-transparent dark background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

    # Border
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (70, 70, 70), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    sq = 12
    for i, (color, label) in enumerate([
        (_GT_COLOR, "Ground truth"),
        (_PRED_COLOR, f"Prediction  conf>={conf_thresh:.2f}"),
    ]):
        row_y = y0 + 22 + i * 28
        cv2.rectangle(img, (x0 + 8, row_y - sq + 2), (x0 + 8 + sq, row_y + 2), color, -1)
        cv2.putText(img, label, (x0 + 26, row_y), font, 0.41, (215, 215, 215), 1, cv2.LINE_AA)


def _make_info_bar(
    img_w: int,
    frame_idx: int,
    total_frames: int,
    gt_count: int,
    pred_count: int,
    conf_thresh: float,
) -> Any:
    import cv2
    import numpy as np

    bar = np.full((_INFO_BAR_H, img_w, 3), 22, dtype=np.uint8)
    # Thin top separator line
    bar[0, :] = (55, 55, 55)
    text = (
        f"YOLOv26x  |  VisDrone val  |  "
        f"Frame {frame_idx + 1:03d} / {total_frames:03d}  |  "
        f"GT: {gt_count}  |  Pred: {pred_count}  |  conf >= {conf_thresh:.2f}"
    )
    cv2.putText(bar, text, (10, _INFO_BAR_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (185, 185, 185), 1, cv2.LINE_AA)
    return bar


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA")
