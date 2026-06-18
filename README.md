# Code Repository

This directory is the structured source repository pushed to GitHub as `distill-nas-yolov26-visdrone`.

## Current Baseline

The default baseline is `YOLOv26x` on the VisDrone validation split. Runtime execution happens on Kaggle, not on the local machine.

## Kaggle Benchmark Command

```bash
visdrone-det benchmark-yolov26x   --data-root /kaggle/input/visdrone-dataset   --work-dir /kaggle/working/distillnas-yolov26x-benchmark   --split val   --model yolo26x.pt   --imgsz 640   --batch 16   --device 0,1   --wandb-project distillNas
```

The command converts VisDrone annotations to YOLO labels, evaluates YOLOv26x with Ultralytics, logs metrics to W&B, and does not save checkpoints.

## Repository Shape

```text
code/
├── README.md
├── pyproject.toml
├── configs/
├── src/visdrone_det/
└── tests/
```

## Boundaries

- Do not run GPU training or full-dataset evaluation locally.
- Use local tests only for lightweight conversion and CLI checks.
- Store checkpoints in W&B artifacts only when an experiment explicitly needs them. This baseline benchmark does not save checkpoints.
