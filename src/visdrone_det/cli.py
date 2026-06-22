"""Command-line entrypoints for VisDrone experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_yolov26x_benchmark
from .distill import run_supernet_distill
from .distill_famo import run_supernet_distill_famo
from .distill_famo_mask import run_supernet_distill_famo_mask
from .finetune import run_yolov26x_finetune
from .infer import run_yolov26x_inference
from .search import run_student_search


def _wandb_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--wandb-project", default="distillNas")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--run-name", default=None)
    return p


def _distill_supernet_parent() -> argparse.ArgumentParser:
    """Shared args for all three distill-supernet* subcommands."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--data-root", type=Path,
        default=Path("/kaggle/input/datasets/banuprasadb/visdrone-dataset"),
    )
    p.add_argument(
        "--teacher-weights", default="yolov26x-visdrone-teacher:best.pt",
        help="Local .pt path or Ultralytics model name for teacher weights",
    )
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--imgsz",         type=int,   default=640)
    p.add_argument("--batch",         type=int,   default=8)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--device",                    default="0,1")
    p.add_argument("--lr0",           type=float, default=0.001)
    p.add_argument("--lrf",           type=float, default=0.01)
    p.add_argument("--weight-decay",  type=float, default=0.0005)
    p.add_argument("--warmup-epochs",   type=int, default=3)
    p.add_argument("--pretrain-epochs", type=int, default=0,
                   help="Epochs of task-loss-only pre-training before KD begins (0 = off)")
    p.add_argument(
        "--pretrained-backbone", default=None, metavar="WEIGHTS",
        help="Pretrained YOLO weights to initialize supernet backbone/neck (e.g. 'yolo26s.pt')",
    )
    p.add_argument("--cache",  action="store_true", default=False)
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume from supernet_last.pt in --work-dir")
    p.add_argument(
        "--checkpoint-interval", type=int, default=5,
        help="Upload supernet_last.pt as W&B checkpoint artifact every N epochs (0 disables)",
    )
    p.add_argument(
        "--wandb-run-id", default=None,
        help="W&B run ID to resume (loaded from checkpoint if --resume used without this flag)",
    )
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visdrone-det")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser(
        "benchmark-yolov26x",
        help="Evaluate YOLOv26x on VisDrone and log metrics to W&B",
        parents=[_wandb_parent()],
    )
    benchmark.add_argument("--data-root", type=Path, default=Path("/kaggle/input/visdrone-dataset"))
    benchmark.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/distillnas-yolov26x-benchmark"))
    benchmark.add_argument("--split", default="val")
    benchmark.add_argument("--model", default="yolo26x.pt")
    benchmark.add_argument("--imgsz", type=int, default=640)
    benchmark.add_argument("--batch", type=int, default=16)
    benchmark.add_argument("--device", default="0,1")

    infer = subparsers.add_parser(
        "infer-yolov26x",
        help="Run YOLOv26x inference on VisDrone val, save annotated video to W&B",
        parents=[_wandb_parent()],
    )
    infer.add_argument("--data-root", type=Path, default=Path("/kaggle/input/datasets/banuprasadb/visdrone-dataset"))
    infer.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/distillnas-yolov26x-infer"))
    infer.add_argument("--split", default="val")
    infer.add_argument("--model", default="yolo26x.pt")
    infer.add_argument("--imgsz", type=int, default=640)
    infer.add_argument("--conf", type=float, default=0.25)
    infer.add_argument("--device", default="0,1")
    infer.add_argument("--max-frames", type=int, default=300)
    infer.add_argument("--fps", type=float, default=5.0)

    finetune = subparsers.add_parser(
        "finetune-yolov26x",
        help="Fine-tune YOLOv26x on VisDrone train split (teacher stage)",
        parents=[_wandb_parent()],
    )
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
                          help="Log last.pt as W&B artifact every N epochs (0 disables)")
    finetune.add_argument("--live-batch-log", action="store_true", default=False,
                          help="Log per-batch training losses to W&B (noisy, for debugging)")
    finetune.add_argument("--wandb-run-id", default=None,
                          help="W&B run ID to resume (uses resume='allow' to continue existing run)")

    distill = subparsers.add_parser(
        "distill-supernet",
        help="Train YOLO supernet student with feature KD from YOLOv26x teacher",
        parents=[_distill_supernet_parent(), _wandb_parent()],
    )
    distill.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/distillnas-supernet-distill"))
    distill.add_argument("--distill-weight", type=float, default=1.0,
                         help="Lambda weight for MSE feature distillation loss")
    distill.add_argument("--task-weight", type=float, default=1.0,
                         help="Lambda weight for detection task loss")

    distill_famo = subparsers.add_parser(
        "distill-supernet-famo",
        help="Train YOLO supernet with FAMO automatic loss weighting (NeurIPS 2023)",
        parents=[_distill_supernet_parent(), _wandb_parent()],
    )
    distill_famo.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/distillnas-supernet-distill-famo"))
    distill_famo.add_argument("--famo-gamma", type=float, default=0.01,
                              help="FAMO weight-update step size (how fast weights adapt to loss changes)")

    distill_famo_mask = subparsers.add_parser(
        "distill-supernet-famo-mask",
        help="SPOS supernet distillation with FAMO + foreground-masked MSE (NeurIPS 2023 + GT heatmap)",
        parents=[_distill_supernet_parent(), _wandb_parent()],
    )
    distill_famo_mask.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/distillnas-supernet-distill-famo-mask"))
    distill_famo_mask.add_argument("--famo-gamma", type=float, default=0.01,
                                   help="FAMO weight-update step size")
    distill_famo_mask.add_argument("--mask-sigma", type=float, default=2.0,
                                   help="Gaussian sigma in grid cells for the foreground heatmap mask")

    search = subparsers.add_parser(
        "search-student",
        help="Search the trained supernet for the best sub-architecture (Stage 3 NAS)",
        parents=[_wandb_parent()],
    )
    search.add_argument("--supernet-weights", default="yolov26x-supernet-student:supernet_best.pt",
                        help="Path to supernet_best.pt from distill-supernet, or W&B artifact ref")
    search.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/distillnas-student-search"))
    search.add_argument("--n-samples", type=int, default=50,
                        help="Number of random sub-architectures to evaluate (random search only)")
    search.add_argument("--search-mode", choices=["random", "exhaustive"], default="random",
                        help="'random': sample n-samples; 'exhaustive': evaluate all 1,296 sub-networks")
    search.add_argument("--imgsz", type=int, default=320)
    search.add_argument("--batch", type=int, default=4)
    search.add_argument("--n-proxy-batches", type=int, default=4,
                        help="Number of random batches to average proxy score over per architecture")
    search.add_argument("--device", default="0")
    search.add_argument("--seed", type=int, default=42)
    search.add_argument("--dummy-weights", action="store_true", default=False,
                        help="Skip loading supernet weights and use random init (for testing)")

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
            pretrain_epochs=args.pretrain_epochs,
            distill_weight=args.distill_weight,
            task_weight=args.task_weight,
            pretrained_backbone=args.pretrained_backbone,
            cache=args.cache,
            resume=args.resume,
            checkpoint_interval=args.checkpoint_interval,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_id=args.wandb_run_id,
            run_name=args.run_name,
        )
        return 0
    if args.command == "distill-supernet-famo":
        run_supernet_distill_famo(
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
            pretrain_epochs=args.pretrain_epochs,
            famo_gamma=args.famo_gamma,
            pretrained_backbone=args.pretrained_backbone,
            cache=args.cache,
            resume=args.resume,
            checkpoint_interval=args.checkpoint_interval,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_id=args.wandb_run_id,
            run_name=args.run_name,
        )
        return 0
    if args.command == "distill-supernet-famo-mask":
        run_supernet_distill_famo_mask(
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
            pretrain_epochs=args.pretrain_epochs,
            famo_gamma=args.famo_gamma,
            mask_sigma=args.mask_sigma,
            pretrained_backbone=args.pretrained_backbone,
            cache=args.cache,
            resume=args.resume,
            checkpoint_interval=args.checkpoint_interval,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_id=args.wandb_run_id,
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
