"""Command-line entrypoints for VisDrone experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_yolov26x_benchmark


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
