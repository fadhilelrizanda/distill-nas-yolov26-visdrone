"""YOLOv26x fine-tuning runner for VisDrone (teacher stage)."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .visdrone import prepare_yolo_dataset, VISDRONE_NAMES


def _make_epoch_callbacks(
    run_id: str,
    project: str,
    entity: str | None,
    checkpoint_interval: int = 5,
) -> tuple[Any, Any]:
    """Return (on_fit_epoch_end, on_model_save) callbacks for DDP-aware per-epoch W&B logging.

    Inside DDP subprocesses wandb.run is None, so each callback re-attaches to the
    parent-process run via resume="must" before logging.
    """
    import wandb

    def _ensure_run() -> None:
        if not wandb.run:
            wandb.init(project=project, entity=entity, id=run_id, resume="must")

    def on_fit_epoch_end(trainer: Any) -> None:
        if getattr(trainer, "rank", 0) != 0:
            return
        _ensure_run()
        epoch = trainer.epoch + 1
        log: dict[str, Any] = {"epoch": epoch, **trainer.metrics}
        if hasattr(trainer, "optimizer") and trainer.optimizer:
            for i, pg in enumerate(trainer.optimizer.param_groups):
                log[f"lr/pg{i}"] = pg["lr"]
        wandb.log(log, step=epoch)

    def on_model_save(trainer: Any) -> None:
        if getattr(trainer, "rank", 0) != 0:
            return
        epoch = trainer.epoch + 1
        if epoch % checkpoint_interval != 0:
            return
        _ensure_run()
        last_pt = Path(trainer.last)
        if not last_pt.exists():
            return
        art = wandb.Artifact(
            name=f"checkpoint-epoch{epoch:04d}",
            type="model",
            metadata={"epoch": epoch, **trainer.metrics},
        )
        art.add_file(str(last_pt), name="last.pt")
        wandb.log_artifact(art)

    return on_fit_epoch_end, on_model_save


def _replay_csv_to_wandb(csv_path: Path) -> None:
    """Replay Ultralytics results.csv to W&B as fallback if callbacks missed epochs."""
    import wandb

    if not csv_path.exists() or not wandb.run:
        return
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Ultralytics pads column names with spaces — strip all keys and values.
            stripped = {k.strip(): v.strip() for k, v in row.items()}
            try:
                epoch = int(float(stripped.pop("epoch", 0)))
            except ValueError:
                continue
            step_metrics: dict[str, Any] = {}
            for k, v in stripped.items():
                try:
                    step_metrics[k] = float(v)
                except ValueError:
                    pass
            if step_metrics:
                wandb.log({"epoch": epoch, **step_metrics}, step=epoch)


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

    cb_epoch, cb_checkpoint = _make_epoch_callbacks(
        run_id=wandb_run.id,
        project=wandb_project,
        entity=wandb_entity,
    )

    metrics: dict[str, Any] = {}
    try:
        # If resume is requested and a previous last.pt exists in results, use it.
        last_pt_resume = results_dir / "yolov26x_visdrone_finetune" / "weights" / "last.pt"
        if resume and last_pt_resume.exists():
            model = YOLO(str(last_pt_resume))
        else:
            model = YOLO(model_name)

        # Replace Ultralytics' built-in W&B callbacks with our DDP-aware versions so
        # there is exactly one W&B run and metrics are logged every epoch in real time.
        for _event in ("on_pretrain_routine_start", "on_fit_epoch_end", "on_val_end", "on_model_save", "on_train_end"):
            model.clear_callback(_event)
        model.add_callback("on_fit_epoch_end", cb_epoch)
        model.add_callback("on_model_save", cb_checkpoint)

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

        # Safety net: replay results.csv in case DDP subprocess callbacks missed epochs.
        _replay_csv_to_wandb(results_dir / "yolov26x_visdrone_finetune" / "results.csv")

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
