"""YOLOv26x fine-tuning runner for VisDrone (teacher stage)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from .patches import install_patches
from .visdrone import prepare_yolo_dataset, VISDRONE_NAMES

# ── background poller for DDP metrics ──────────────────────────────────────


def _poll_metrics(
    metrics_file: Path,
    poll_interval: float,
    stop_event: threading.Event,
) -> Iterator[dict[str, Any]]:
    """Yield events from the JSONL *metrics_file* as they arrive.

    Runs in a background thread while ``model.train()`` blocks the parent
    process.  Yields parsed dicts — the caller decides what to do with them.
    """
    last_pos = 0
    # Wait for the file to exist (DDP subprocess may take a moment to start).
    for _ in range(60):  # up to 60 s
        if stop_event.is_set() or metrics_file.exists():
            break
        time.sleep(1)
    while not stop_event.is_set():
        try:
            with open(metrics_file) as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
                last_pos = f.tell()
        except FileNotFoundError:
            pass
        except Exception:
            pass
        time.sleep(poll_interval)


def _wandb_poller_thread(
    metrics_file: Path,
    wandb_run: Any,
    stop_event: threading.Event,
    checkpoint_interval: int,
) -> None:
    """Background thread: polls *metrics_file* and forwards events to W&B.

    Handles three event types:
        ``epoch_end``   → ``wandb.log`` with metrics + LR
        ``checkpoint``  → ``wandb.log_artifact`` (last.pt)
        ``train_end``   → no action (parent handles finalisation)
    """
    import wandb

    for event in _poll_metrics(metrics_file, poll_interval=5.0, stop_event=stop_event):
        evt = event.get("event")
        if evt == "epoch_end":
            epoch = event.get("epoch", 0)
            log = {"epoch": epoch}
            if event.get("metrics"):
                log.update(event["metrics"])
            if event.get("lr"):
                log.update(event["lr"])
            wandb.log(log, step=epoch)
        elif evt == "checkpoint" and checkpoint_interval > 0:
            ckpt_path = event.get("path")
            epoch = event.get("epoch", 0)
            if ckpt_path and Path(ckpt_path).exists():
                art = wandb.Artifact(
                    name=f"checkpoint-epoch{epoch:04d}",
                    type="model",
                    metadata={"epoch": epoch, **(event.get("metrics") or {})},
                )
                art.add_file(ckpt_path, name="last.pt")
                wandb.log_artifact(art)
        # train_end is a no-op; the parent's finally block handles finish()


# ── main entrypoint ────────────────────────────────────────────────────────


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
    save_period: int = -1,
    checkpoint_interval: int = 0,
    wandb_project: str = "distillNas",
    wandb_entity: str | None = None,
    wandb_run_id: str | None = None,
    run_name: str | None = None,
    live_batch_log: bool = False,
) -> dict[str, Any]:
    """Fine-tune YOLOv26x on VisDrone train split and evaluate on val, logging to W&B.

    Per-epoch metrics and checkpoints are logged to W&B in real time via
    a background thread that reads from a shared JSONL file written by the
    DDP subprocess (see :mod:`patches`).

    Parameters
    ----------
    live_batch_log:
        If True, log per-batch training losses to W&B (noisy — useful for
        debugging convergence issues, not for routine training).
    """
    from ultralytics import YOLO
    import wandb

    work_dir = work_dir.expanduser().resolve()
    dataset_dir = work_dir / "visdrone_yolo"
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = results_dir / "ddp_metrics.jsonl"

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

    wandb_init_kwargs: dict[str, Any] = dict(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        job_type="finetune",
        tags=["visdrone", "yolov26x", "finetune", "teacher", "kaggle"],
        config=config,
    )
    if wandb_run_id:
        wandb_init_kwargs["resume"] = "allow"
        wandb_init_kwargs["id"] = wandb_run_id
    wandb_run = wandb.init(**wandb_init_kwargs)

    # ------------------------------------------------------------------
    # DDP-aware W&B logging
    #
    # Ultralytics DDP spawns a subprocess (torch.distributed.run) that
    # re-imports the trainer from scratch and loses the parent's W&B run
    # and custom callbacks.  We cannot call wandb.init() from inside the
    # DDP subprocess — it times out after 90 s (CommError).
    #
    # Instead, the monkey-patch writes per-epoch / per-checkpoint events
    # to a shared JSONL file.  A background thread in the parent polls
    # this file and forwards events to W&B over the already-established
    # connection.
    #
    # Environment variables survive subprocess.run() and are the bridge.
    # ------------------------------------------------------------------
    os.environ["_WANDB_METRICS_FILE"] = str(metrics_file)
    install_patches(checkpoint_interval=checkpoint_interval, live_batch_log=live_batch_log)

    # Clean slate for metrics file.
    metrics_file.write_text("", encoding="utf-8")

    stop_event = threading.Event()
    poller_thread = threading.Thread(
        target=_wandb_poller_thread,
        args=(metrics_file, wandb_run, stop_event, checkpoint_interval),
        daemon=True,
    )
    poller_thread.start()

    metrics: dict[str, Any] = {}
    try:
        # Resume checkpoint lookup: check a dedicated dir first (outside Ultralytics save_dir
        # so trainer init can't accidentally remove it), then fall back to weights dir.
        resume_ckpt = work_dir / "resume_checkpoint" / "last.pt"
        weights_ckpt = results_dir / "yolov26x_visdrone_finetune" / "weights" / "last.pt"
        if resume and resume_ckpt.exists():
            model = YOLO(str(resume_ckpt))
        elif resume and weights_ckpt.exists():
            model = YOLO(str(weights_ckpt))
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
            save_period=save_period,
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
        stop_event.set()
        poller_thread.join(timeout=30)
        wandb.finish()

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def _extract_metrics(results: Any) -> dict[str, Any]:
    box = getattr(results, "box", None)
    speed = getattr(results, "speed", None) or {}
    extracted = {
        "metrics/mAP50-95(B)": _as_float(getattr(box, "map", None)),
        "metrics/mAP50(B)": _as_float(getattr(box, "map50", None)),
        "metrics/mAP75(B)": _as_float(getattr(box, "map75", None)),
        "metrics/precision(B)": _as_float(getattr(box, "mp", None)),
        "metrics/recall(B)": _as_float(getattr(box, "mr", None)),
    }
    for key, value in speed.items():
        extracted[f"speed/{key}_ms"] = _as_float(value)
    return {k: v for k, v in extracted.items() if v is not None}


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
