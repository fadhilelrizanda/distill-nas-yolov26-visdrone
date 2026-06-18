"""YOLOv26x baseline benchmark runner for Kaggle."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .visdrone import prepare_yolo_dataset


def run_yolov26x_benchmark(
    data_root: Path,
    work_dir: Path,
    split: str = "val",
    model_name: str = "yolo26x.pt",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0,1",
    wandb_project: str = "distillNas",
    wandb_entity: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Prepare VisDrone and evaluate YOLOv26x with W&B metric logging."""
    from ultralytics import YOLO
    import wandb

    work_dir = work_dir.expanduser().resolve()
    dataset_dir = work_dir / "visdrone_yolo"
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_yolo_dataset(data_root=data_root, output_root=dataset_dir, split=split)
    git_sha = _git_sha(Path.cwd())
    run_name = run_name or f"baseline/yolov26x/visdrone-{split}/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        job_type="baseline-eval",
        tags=["visdrone", "yolov26x", "baseline", "eval", "kaggle", "no-checkpoint"],
        config={
            "model": model_name,
            "baseline_model": "YOLOv26x",
            "dataset": "VisDrone",
            "kaggle_dataset_slug": "banuprasadb/visdrone-dataset",
            "split": split,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "accelerator": "NvidiaTeslaT4",
            "expected_gpu_count": 2,
            "checkpoint_saving": "disabled",
            "git_sha": git_sha,
            "python": platform.python_version(),
        },
    )

    metrics: dict[str, Any]
    try:
        model = YOLO(model_name)
        results = model.val(
            data=str(prepared.yaml_path),
            split="val",
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(results_dir),
            name="yolov26x_visdrone_val",
            exist_ok=True,
            save=False,
            save_json=False,
            plots=False,
            verbose=True,
        )
        metrics = _extract_metrics(results)
        metrics.update(
            {
                "dataset/image_count": prepared.image_count,
                "dataset/label_count": prepared.label_count,
                "dataset/skipped_box_count": prepared.skipped_box_count,
                "run/git_sha": git_sha,
            }
        )
        wandb.log(metrics)
        wandb_run.summary.update(metrics)
        metrics_path = results_dir / "yolov26x_visdrone_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        wandb.save(str(metrics_path), policy="now")
    finally:
        wandb.finish()

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def _extract_metrics(results: Any) -> dict[str, Any]:
    box = getattr(results, "box", None)
    speed = getattr(results, "speed", None) or {}
    metrics = {
        "metrics/mAP50-95(B)": _as_float(getattr(box, "map", None)),
        "metrics/mAP50(B)": _as_float(getattr(box, "map50", None)),
        "metrics/mAP75(B)": _as_float(getattr(box, "map75", None)),
        "metrics/precision(B)": _as_float(getattr(box, "mp", None)),
        "metrics/recall(B)": _as_float(getattr(box, "mr", None)),
    }
    for key, value in speed.items():
        metrics[f"speed/{key}_ms"] = _as_float(value)
    return {key: value for key, value in metrics.items() if value is not None}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA")
