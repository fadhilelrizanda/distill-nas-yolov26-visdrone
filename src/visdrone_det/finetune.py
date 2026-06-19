"""YOLOv26x fine-tuning runner for VisDrone (teacher stage)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .visdrone import prepare_yolo_dataset, VISDRONE_NAMES


def run_yolov26x_finetune(
    data_root: Path,
    work_dir: Path,
    model_name: str = "yolo26x.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 4,
    device: str = "0,1",
    patience: int = 20,
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    warmup_epochs: int = 3,
    cache: bool = False,
    resume: bool = False,
    wandb_project: str = "distillNas",
    wandb_entity: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Fine-tune YOLOv26x on VisDrone train split and evaluate on val, logging to W&B."""
    from ultralytics import YOLO
    import wandb

    work_dir = work_dir.expanduser().resolve()
    dataset_dir = work_dir / "visdrone_yolo"
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prepare both splits; each call creates images/<split>/ and labels/<split>/ symlinks.
    train_prepared = prepare_yolo_dataset(data_root=data_root, output_root=dataset_dir, split="train")
    val_prepared = prepare_yolo_dataset(data_root=data_root, output_root=dataset_dir, split="val")

    # Write a combined YAML that points train and val to their respective split dirs.
    combined_yaml_path = dataset_dir / "visdrone_finetune.yaml"
    combined_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "images/train",
                "val": "images/val",
                "nc": len(VISDRONE_NAMES),
                "names": {idx: name for idx, name in enumerate(VISDRONE_NAMES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    git_sha = _git_sha(Path.cwd())
    run_name = run_name or "yolov26x-finetune"

    config = {
        "model": model_name,
        "baseline_model": "YOLOv26x",
        "dataset": "VisDrone",
        "kaggle_dataset_slug": "banuprasadb/visdrone-dataset",
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device,
        "patience": patience,
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "cache": cache,
        "resume": resume,
        "accelerator": "NvidiaTeslaT4",
        "expected_gpu_count": 2,
        "git_sha": git_sha,
        "python": platform.python_version(),
    }

    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        job_type="finetune",
        tags=["visdrone", "yolov26x", "finetune", "teacher", "kaggle"],
        config=config,
    )

    metrics: dict[str, Any] = {}
    try:
        # If resume is requested and a previous last.pt exists in results, use it.
        last_pt_resume = results_dir / "yolov26x_visdrone_finetune" / "weights" / "last.pt"
        if resume and last_pt_resume.exists():
            model = YOLO(str(last_pt_resume))
        else:
            model = YOLO(model_name)

        model.train(
            data=str(combined_yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=device,
            patience=patience,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            cache=cache,
            project=str(results_dir),
            name="yolov26x_visdrone_finetune",
            exist_ok=True,
            save=True,
            plots=True,
            verbose=True,
        )

        best_pt = results_dir / "yolov26x_visdrone_finetune" / "weights" / "best.pt"
        last_pt = results_dir / "yolov26x_visdrone_finetune" / "weights" / "last.pt"

        # Evaluate best checkpoint on val split.
        val_model = YOLO(str(best_pt)) if best_pt.exists() else model
        val_results = val_model.val(
            data=str(combined_yaml_path),
            split="val",
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(results_dir),
            name="yolov26x_visdrone_val_final",
            exist_ok=True,
            save=False,
            save_json=False,
            plots=True,
            verbose=True,
        )

        metrics = _extract_metrics(val_results)
        metrics.update(
            {
                "dataset/train_image_count": train_prepared.image_count,
                "dataset/val_image_count": val_prepared.image_count,
                "dataset/train_label_count": train_prepared.label_count,
                "dataset/val_label_count": val_prepared.label_count,
                "run/git_sha": git_sha,
            }
        )
        wandb.log(metrics)
        wandb_run.summary.update(metrics)

        metrics_path = results_dir / "yolov26x_finetune_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        wandb.save(str(metrics_path), policy="now")

        # Upload best.pt (and last.pt) as a W&B model artifact.
        if best_pt.exists():
            artifact = wandb.Artifact(
                name="yolov26x-visdrone-teacher",
                type="model",
                description="YOLOv26x fine-tuned on VisDrone (teacher checkpoint for KD/NAS)",
                metadata={**config, **metrics},
            )
            artifact.add_file(str(best_pt), name="best.pt")
            if last_pt.exists():
                artifact.add_file(str(last_pt), name="last.pt")
            wandb_run.log_artifact(artifact)

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
    return {k: v for k, v in metrics.items() if v is not None}


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
