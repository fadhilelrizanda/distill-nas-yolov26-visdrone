"""YOLOv26x inference visualization runner for Kaggle."""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .visdrone import prepare_yolo_dataset


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
    """Run YOLOv26x predict on VisDrone val images, write annotated video, upload to W&B."""
    import cv2
    from ultralytics import YOLO
    import wandb

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    work_dir = work_dir.expanduser().resolve()
    dataset_dir = work_dir / "visdrone_yolo"
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_yolo_dataset(data_root=data_root, output_root=dataset_dir, split=split)
    git_sha = _git_sha(Path.cwd())
    run_name = run_name or f"yolov26x-infer"

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
            "accelerator": "NvidiaTeslaT4",
            "git_sha": git_sha,
            "python": platform.python_version(),
        },
    )

    video_path = results_dir / "yolov26x_visdrone_inference.mp4"

    try:
        model = YOLO(model_name)

        # Determine frame size from first image
        first = cv2.imread(str(image_paths[0]))
        if first is None:
            raise RuntimeError(f"Could not read first image: {image_paths[0]}")
        h, w = first.shape[:2]

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
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
            frame = result.plot()  # BGR numpy array with drawn boxes and labels
            writer.write(frame)
            written += 1
            if written % 50 == 0:
                print(f"  wrote {written}/{frame_count} frames", flush=True)

        writer.release()
        print(f"Video saved: {video_path} ({written} frames)", flush=True)

        wandb_run.log({
            "inference/video": wandb.Video(str(video_path), fps=fps, format="mp4"),
            "inference/frame_count": written,
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
                "git_sha": git_sha,
            },
        )
        art.add_file(str(video_path))
        wandb_run.log_artifact(art)

    finally:
        wandb.finish()

    return video_path


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA")
