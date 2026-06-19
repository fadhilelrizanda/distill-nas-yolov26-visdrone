"""Command-line entrypoints for VisDrone experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_yolov26x_benchmark
from .finetune import run_yolov26x_finetune
from .infer import run_yolov26x_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visdrone-det")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark-yolov26x", help="Evaluate YOLOv26x on VisDrone and log metrics to W&B")
    benchmark.add_argument("--data-root", type=Path, default=Path("/kaggle/input/visdrone-dataset"))
    benchmark.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/distillnas-yolov26x-benchmark"))
    benchmark.add_argument("--split", default="val")
    benchmark.add_argument("--model", default="yolo26x.pt")
    benchmark.add_argument("--imgsz", type=int, default=640)
    benchmark.add_argument("--batch", type=int, default=16)
    benchmark.add_argument("--device", default="0,1")
    benchmark.add_argument("--wandb-project", default="distillNas")
    benchmark.add_argument("--wandb-entity", default=None)
    benchmark.add_argument("--run-name", default=None)

    infer = subparsers.add_parser("infer-yolov26x", help="Run YOLOv26x inference on VisDrone val, save annotated video to W&B")
    infer.add_argument("--data-root", type=Path, default=Path("/kaggle/input/datasets/banuprasadb/visdrone-dataset"))
    infer.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/distillnas-yolov26x-infer"))
    infer.add_argument("--split", default="val")
    infer.add_argument("--model", default="yolo26x.pt")
    infer.add_argument("--imgsz", type=int, default=640)
    infer.add_argument("--conf", type=float, default=0.25)
    infer.add_argument("--device", default="0")
    infer.add_argument("--max-frames", type=int, default=300)
    infer.add_argument("--fps", type=float, default=5.0)
    infer.add_argument("--wandb-project", default="distillNas")
    infer.add_argument("--wandb-entity", default=None)
    infer.add_argument("--run-name", default=None)

    finetune = subparsers.add_parser("finetune-yolov26x", help="Fine-tune YOLOv26x on VisDrone train split (teacher stage)")
    finetune.add_argument("--data-root", type=Path, default=Path("/kaggle/input/datasets/banuprasadb/visdrone-dataset"))
    finetune.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/distillnas-yolov26x-finetune"))
    finetune.add_argument("--model", default="yolo26x.pt")
    finetune.add_argument("--epochs", type=int, default=100)
    finetune.add_argument("--imgsz", type=int, default=640)
    finetune.add_argument("--batch", type=int, default=8)
    finetune.add_argument("--workers", type=int, default=4)
    finetune.add_argument("--device", default="0,1")
    finetune.add_argument("--patience", type=int, default=20)
    finetune.add_argument("--optimizer", default="AdamW")
    finetune.add_argument("--lr0", type=float, default=0.001)
    finetune.add_argument("--lrf", type=float, default=0.01)
    finetune.add_argument("--weight-decay", type=float, default=0.0005)
    finetune.add_argument("--warmup-epochs", type=int, default=3)
    finetune.add_argument("--cache", action="store_true", default=False)
    finetune.add_argument("--resume", action="store_true", default=False)
    finetune.add_argument("--wandb-project", default="distillNas")
    finetune.add_argument("--wandb-entity", default=None)
    finetune.add_argument("--run-name", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "infer-yolov26x":
        run_yolov26x_inference(
            data_root=args.data_root,
            work_dir=args.work_dir,
            split=args.split,
            model_name=args.model,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            max_frames=args.max_frames,
            fps=args.fps,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            run_name=args.run_name,
        )
        return 0
    if args.command == "benchmark-yolov26x":
        run_yolov26x_benchmark(
            data_root=args.data_root,
            work_dir=args.work_dir,
            split=args.split,
            model_name=args.model,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            run_name=args.run_name,
        )
        return 0
    if args.command == "finetune-yolov26x":
        run_yolov26x_finetune(
            data_root=args.data_root,
            work_dir=args.work_dir,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            cache=args.cache,
            resume=args.resume,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            run_name=args.run_name,
        )
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
