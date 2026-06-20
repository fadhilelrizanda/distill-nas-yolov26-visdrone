"""Command-line entrypoints for VisDrone experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_yolov26x_benchmark
from .distill import run_supernet_distill
from .finetune import run_yolov26x_finetune
from .infer import run_yolov26x_inference
from .search import run_student_search


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
    infer.add_argument("--device", default="0,1")
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
    finetune.add_argument("--save-period", type=int, default=-1,
                         help="Save epoch{N}.pt every N epochs (-1 disables, saves only last.pt/best.pt)")
    finetune.add_argument("--checkpoint-interval", type=int, default=0,
                         help="Log last.pt as W&B artifact every N epochs (0 disables; avoids disk fill from artifact cache)")
    finetune.add_argument("--live-batch-log", action="store_true", default=False,
                         help="Log per-batch training losses to W&B (noisy, for debugging)")
    finetune.add_argument("--wandb-project", default="distillNas")
    finetune.add_argument("--wandb-entity", default=None)
    finetune.add_argument("--wandb-run-id", default=None,
                         help="W&B run ID to resume (uses resume='allow' to continue existing run)")
    finetune.add_argument("--run-name", default=None)

    distill = subparsers.add_parser(
        "distill-supernet",
        help="Train YOLO supernet student with feature KD from YOLOv26x teacher",
    )
    distill.add_argument(
        "--data-root",
        type=Path,
        default=Path("/kaggle/input/datasets/banuprasadb/visdrone-dataset"),
    )
    distill.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/kaggle/working/distillnas-supernet-distill"),
    )
    distill.add_argument(
        "--teacher-weights",
        default="yolov26x-visdrone-teacher:best.pt",
        help="Local .pt path or Ultralytics model name for teacher weights",
    )
    distill.add_argument("--epochs", type=int, default=50)
    distill.add_argument("--imgsz", type=int, default=640)
    distill.add_argument("--batch", type=int, default=8)
    distill.add_argument("--workers", type=int, default=4)
    distill.add_argument("--device", default="0,1")
    distill.add_argument("--lr0", type=float, default=0.001)
    distill.add_argument("--lrf", type=float, default=0.01)
    distill.add_argument("--weight-decay", type=float, default=0.0005)
    distill.add_argument("--warmup-epochs", type=int, default=3)
    distill.add_argument(
        "--distill-weight",
        type=float,
        default=1.0,
        help="Lambda weight for MSE feature distillation loss",
    )
    distill.add_argument(
        "--task-weight",
        type=float,
        default=1.0,
        help="Lambda weight for detection task loss",
    )
    distill.add_argument(
        "--pretrained-backbone",
        default=None,
        metavar="WEIGHTS",
        help=(
            "Pretrained YOLO weights to initialize the supernet backbone/neck "
            "(e.g. 'yolo26s.pt'). Skipped when --resume is used. "
            "Layers whose shapes differ from the pretrained model are kept random."
        ),
    )
    distill.add_argument("--cache", action="store_true", default=False)
    distill.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from supernet_last.pt in --work-dir",
    )
    distill.add_argument("--wandb-project", default="distillNas")
    distill.add_argument("--wandb-entity", default=None)
    distill.add_argument("--run-name", default=None)

    search = subparsers.add_parser(
        "search-student",
        help="Search the trained supernet for the best sub-architecture (Stage 3 NAS)",
    )
    search.add_argument(
        "--supernet-weights",
        default="yolov26x-supernet-student:supernet_best.pt",
        help="Path to supernet_best.pt from distill-supernet, or W&B artifact ref",
    )
    search.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/kaggle/working/distillnas-student-search"),
    )
    search.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of random sub-architectures to evaluate (random search only)",
    )
    search.add_argument(
        "--search-mode",
        choices=["random", "exhaustive"],
        default="random",
        help="'random': sample n-samples; 'exhaustive': evaluate all 1,296 sub-networks",
    )
    search.add_argument("--imgsz", type=int, default=320)
    search.add_argument("--batch", type=int, default=4)
    search.add_argument(
        "--n-proxy-batches",
        type=int,
        default=4,
        help="Number of random batches to average proxy score over per architecture",
    )
    search.add_argument("--device", default="0")
    search.add_argument("--seed", type=int, default=42)
    search.add_argument(
        "--dummy-weights",
        action="store_true",
        default=False,
        help="Skip loading supernet weights and use random init (for testing)",
    )
    search.add_argument("--wandb-project", default="distillNas")
    search.add_argument("--wandb-entity", default=None)
    search.add_argument("--run-name", default=None)

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
            save_period=args.save_period,
            checkpoint_interval=args.checkpoint_interval,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_id=args.wandb_run_id,
            run_name=args.run_name,
            live_batch_log=args.live_batch_log,
        )
        return 0
    if args.command == "distill-supernet":
        run_supernet_distill(
            data_root=args.data_root,
            work_dir=args.work_dir,
            teacher_weights=args.teacher_weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            lr0=args.lr0,
            lrf=args.lrf,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            distill_weight=args.distill_weight,
            task_weight=args.task_weight,
            pretrained_backbone=args.pretrained_backbone,
            cache=args.cache,
            resume=args.resume,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            run_name=args.run_name,
        )
        return 0
    if args.command == "search-student":
        run_student_search(
            supernet_weights=args.supernet_weights,
            work_dir=args.work_dir,
            n_samples=args.n_samples,
            search_mode=args.search_mode,
            imgsz=args.imgsz,
            batch=args.batch,
            n_proxy_batches=args.n_proxy_batches,
            device=args.device,
            seed=args.seed,
            dummy_weights=args.dummy_weights,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            run_name=args.run_name,
        )
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
