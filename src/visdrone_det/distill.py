"""Supernet student distillation trainer for VisDrone."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from .supernet import (
    BACKBONE_DEPTH_CHOICES,
    NECK_DEPTH_CHOICES,
    ArchConfig,
    YOLOSupernet,
    _DEFAULT_T_P3,
    _DEFAULT_T_P4,
    _DEFAULT_T_P5,
)
from .visdrone import VISDRONE_NAMES, prepare_yolo_dataset


# ── Dataset ────────────────────────────────────────────────────────────────


class _VisDroneDataset(Dataset):
    """Minimal YOLO-format image + label loader for VisDrone splits.

    Reads from the symlinked ``images/`` and ``labels/`` directories produced
    by ``prepare_yolo_dataset()``.  Applies letterbox resize to a fixed square.
    Returns ``(img_tensor [3, H, W], labels [N, 5])`` where labels are
    ``[class_id, cx, cy, w, h]`` in [0, 1] range.
    """

    def __init__(self, images_dir: Path, labels_dir: Path, imgsz: int) -> None:
        self.imgsz = imgsz
        self.labels_dir = labels_dir
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.images: list[Path] = sorted(
            p for ext in exts for p in images_dir.glob(ext)
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_t, scale, pad_left, pad_top = _letterbox(img, self.imgsz)

        labels: list[list[float]] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                cx_n, cy_n, w_n, h_n = (float(p) for p in parts[1:])
                # Adjust normalized coords for letterbox transform
                cx_px = cx_n * orig_w * scale + pad_left
                cy_px = cy_n * orig_h * scale + pad_top
                w_px = w_n * orig_w * scale
                h_px = h_n * orig_h * scale
                # Re-normalize to imgsz
                cx_r = cx_px / self.imgsz
                cy_r = cy_px / self.imgsz
                w_r = w_px / self.imgsz
                h_r = h_px / self.imgsz
                if 0 < cx_r < 1 and 0 < cy_r < 1 and w_r > 0 and h_r > 0:
                    labels.append([cls, cx_r, cy_r, w_r, h_r])

        labels_t = (
            torch.tensor(labels, dtype=torch.float32)
            if labels
            else torch.zeros((0, 5), dtype=torch.float32)
        )
        return img_t, labels_t


def _letterbox(
    img: Image.Image,
    size: int,
) -> tuple[torch.Tensor, float, int, int]:
    """Resize and pad image to ``size × size`` (letterbox).

    Returns ``(tensor [3, size, size], scale, pad_left, pad_top)``.
    """
    w, h = img.size
    scale = min(size / w, size / h)
    new_w = int(w * scale + 0.5)
    new_h = int(h * scale + 0.5)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    canvas.paste(img, (pad_left, pad_top))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # HWC → CHW
    return tensor, scale, pad_left, pad_top


def _collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack images; prepend per-image batch index to label rows."""
    imgs, labels_list = zip(*batch)
    imgs_t = torch.stack(imgs)
    tagged: list[torch.Tensor] = []
    for i, labels in enumerate(labels_list):
        if len(labels) > 0:
            idx_col = torch.full((len(labels), 1), float(i))
            tagged.append(torch.cat([idx_col, labels], dim=1))
    targets = (
        torch.cat(tagged, dim=0)
        if tagged
        else torch.zeros((0, 6), dtype=torch.float32)
    )
    return imgs_t, targets


# ── Teacher FPN probe and hooks ────────────────────────────────────────────

# Module-level store: populated by teacher forward hooks, read by _distill_loss().
_TEACHER_FEAT_STORE: dict[str, torch.Tensor] = {}


def _find_fpn_layers(
    teacher_nn: nn.Module,
    imgsz: int = 640,
) -> tuple[dict[str, nn.Module], dict[str, int]]:
    """Locate teacher FPN output modules at P3/P4/P5 by shape probing.

    Hooks every direct child of ``teacher_nn.model`` (the Ultralytics
    nn.Sequential), runs a dummy forward on CPU, then picks the LAST module
    to output at each of (H/8, W/8), (H/16, W/16), (H/32, W/32).
    "Last wins" ensures we capture neck outputs, not backbone laterals that
    share the same spatial dimensions.

    Returns
    -------
    modules : dict mapping "P3"/"P4"/"P5" → nn.Module
    channels : dict mapping "P3"/"P4"/"P5" → int (output channel count)
    """
    layers = list(teacher_nn.model.children())
    # (module, shape) for the last module seen at each spatial resolution
    last_at: dict[tuple[int, int], tuple[nn.Module, int]] = {}
    hooks: list[Any] = []

    for layer in layers:
        def _make_hook(mod: nn.Module) -> Any:
            def hook(m: nn.Module, inp: Any, out: Any) -> None:
                if isinstance(out, torch.Tensor) and out.dim() == 4:
                    hw = (out.shape[2], out.shape[3])
                    last_at[hw] = (mod, out.shape[1])

            return hook

        hooks.append(layer.register_forward_hook(_make_hook(layer)))

    was_training = teacher_nn.training
    teacher_nn.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    try:
        with torch.no_grad():
            teacher_nn(dummy)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            teacher_nn.train()

    target_hw = {
        (imgsz // 8, imgsz // 8): "P3",
        (imgsz // 16, imgsz // 16): "P4",
        (imgsz // 32, imgsz // 32): "P5",
    }

    modules: dict[str, nn.Module] = {}
    channels: dict[str, int] = {}
    for hw, (mod, ch) in last_at.items():
        if hw in target_hw:
            key = target_hw[hw]
            modules[key] = mod
            channels[key] = ch

    missing = {"P3", "P4", "P5"} - set(modules.keys())
    if missing:
        found_shapes = sorted(last_at.keys())
        raise RuntimeError(
            f"Could not locate teacher FPN layers for {missing}. "
            f"Shapes found at: {found_shapes}"
        )
    return modules, channels


def _register_teacher_hooks(fpn_modules: dict[str, nn.Module]) -> list:
    """Attach forward hooks that populate ``_TEACHER_FEAT_STORE`` during teacher forward."""
    handles: list[Any] = []
    for key, module in fpn_modules.items():
        def _make_hook(k: str) -> Any:
            def hook(m: nn.Module, inp: Any, out: torch.Tensor) -> None:
                _TEACHER_FEAT_STORE[k] = out.detach()

            return hook

        handles.append(module.register_forward_hook(_make_hook(key)))
    return handles


# ── Loss functions ─────────────────────────────────────────────────────────


def _distill_loss(
    student_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Average MSE across P3/P4/P5 between projected student and teacher features."""
    total = student_feats[0].new_tensor(0.0)
    count = 0
    for s_feat, key in zip(student_feats, ("P3", "P4", "P5")):
        t_feat = _TEACHER_FEAT_STORE.get(key)
        if t_feat is None:
            continue
        total = total + F.mse_loss(s_feat, t_feat.to(s_feat.device))
        count += 1
    return total / max(1, count)


def _giou(
    pred: torch.Tensor,  # [N, 4] in x1y1x2y2 pixel coords
    gt: torch.Tensor,    # [N, 4]
) -> torch.Tensor:
    """GIoU loss (scalar mean)."""
    # Intersection
    ix1 = torch.max(pred[:, 0], gt[:, 0])
    iy1 = torch.max(pred[:, 1], gt[:, 1])
    ix2 = torch.min(pred[:, 2], gt[:, 2])
    iy2 = torch.min(pred[:, 3], gt[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    pred_area = ((pred[:, 2] - pred[:, 0]).clamp(min=0)
                 * (pred[:, 3] - pred[:, 1]).clamp(min=0))
    gt_area = ((gt[:, 2] - gt[:, 0]).clamp(min=0)
               * (gt[:, 3] - gt[:, 1]).clamp(min=0))
    union = pred_area + gt_area - inter + 1e-7
    iou = inter / union

    # Enclosing box
    ex1 = torch.min(pred[:, 0], gt[:, 0])
    ey1 = torch.min(pred[:, 1], gt[:, 1])
    ex2 = torch.max(pred[:, 2], gt[:, 2])
    ey2 = torch.max(pred[:, 3], gt[:, 3])
    enc_area = ((ex2 - ex1) * (ey2 - ey1)).clamp(min=1e-7)

    giou_val = iou - (enc_area - union) / enc_area
    return (1.0 - giou_val).mean()


class _TaskLoss:
    """Simplified anchor-free detection loss for supernet training regularization.

    Uses area-based scale assignment and decodes predicted boxes with sigmoid
    centering + exp sizing.  Intentionally simple — the primary learning signal
    comes from feature distillation; this keeps the backbone grounded in GT.
    """

    _STRIDES = [8, 16, 32]
    # Box area (px²) thresholds at imgsz=640: P3 < 1024 <= P4 < 9216 <= P5
    _AREA_THR = [32**2, 96**2]

    def __init__(self, num_classes: int = 10) -> None:
        self.nc = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")

    def _assign_scale(self, w_n: float, h_n: float, imgsz: int) -> int:
        area = (w_n * imgsz) * (h_n * imgsz)
        if area < self._AREA_THR[0]:
            return 0
        if area < self._AREA_THR[1]:
            return 1
        return 2

    def __call__(
        self,
        preds: list[tuple[torch.Tensor, torch.Tensor]],
        targets: torch.Tensor,   # [N, 6]: (batch_idx, cls, cx_n, cy_n, w_n, h_n)
        imgsz: int,
    ) -> torch.Tensor:
        device = preds[0][0].device
        total_cls = preds[0][0].new_tensor(0.0)
        total_box = preds[0][0].new_tensor(0.0)
        num_fg = 0

        for scale_idx, (stride, (cls_logits, box_deltas)) in enumerate(
            zip(self._STRIDES, preds)
        ):
            grid = imgsz // stride
            B = cls_logits.shape[0]
            cls_tgt = torch.zeros_like(cls_logits)  # [B, nc, grid, grid]
            pred_boxes: list[torch.Tensor] = []
            gt_boxes: list[torch.Tensor] = []

            for b in range(B):
                row_mask = targets[:, 0] == b
                for row in targets[row_mask]:
                    _, cls_id, cx_n, cy_n, w_n, h_n = row.tolist()
                    if self._assign_scale(w_n, h_n, imgsz) != scale_idx:
                        continue

                    gi = max(0, min(int(cx_n * grid), grid - 1))
                    gj = max(0, min(int(cy_n * grid), grid - 1))
                    cls_tgt[b, int(cls_id), gj, gi] = 1.0
                    num_fg += 1

                    # GT box in pixels (x1y1x2y2)
                    gt_x1 = (cx_n - w_n / 2) * imgsz
                    gt_y1 = (cy_n - h_n / 2) * imgsz
                    gt_x2 = (cx_n + w_n / 2) * imgsz
                    gt_y2 = (cy_n + h_n / 2) * imgsz

                    # Decode predicted box at assigned cell
                    dx = box_deltas[b, 0, gj, gi]
                    dy = box_deltas[b, 1, gj, gi]
                    dw = box_deltas[b, 2, gj, gi]
                    dh = box_deltas[b, 3, gj, gi]

                    pred_cx = (gi + torch.sigmoid(dx)) * stride
                    pred_cy = (gj + torch.sigmoid(dy)) * stride
                    pred_w = torch.exp(dw.clamp(max=4.0)) * stride
                    pred_h = torch.exp(dh.clamp(max=4.0)) * stride

                    pred_boxes.append(torch.stack([
                        pred_cx - pred_w / 2, pred_cy - pred_h / 2,
                        pred_cx + pred_w / 2, pred_cy + pred_h / 2,
                    ]))
                    gt_boxes.append(
                        torch.tensor([gt_x1, gt_y1, gt_x2, gt_y2], device=device)
                    )

            total_cls = total_cls + self.bce(cls_logits, cls_tgt)
            if pred_boxes:
                total_box = total_box + _giou(
                    torch.stack(pred_boxes),
                    torch.stack(gt_boxes),
                )

        normalizer = float(max(1, num_fg))
        return (total_cls + total_box) / normalizer


# ── Main entrypoint ────────────────────────────────────────────────────────


def run_supernet_distill(
    data_root: Path,
    work_dir: Path,
    teacher_weights: str = "yolov26x-visdrone-teacher:best.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 4,
    device: str = "0,1",
    lr0: float = 0.001,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    warmup_epochs: int = 3,
    distill_weight: float = 1.0,
    task_weight: float = 1.0,
    pretrained_backbone: str | None = None,
    cache: bool = False,
    resume: bool = False,
    wandb_project: str = "distillNas",
    wandb_entity: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train the YOLO supernet student with feature distillation from a teacher.

    Implements Single-Path One-Shot (SPOS): one random sub-architecture is
    sampled and activated per step (mini-batch).  Total loss =
    task_weight * L_task + distill_weight * L_distill, where L_distill is MSE
    between student and teacher FPN features at P3/P4/P5 scales.

    Parameters
    ----------
    teacher_weights:
        Path to a ``.pt`` file (YOLOv26x fine-tuned on VisDrone) or a model
        name resolvable by Ultralytics (e.g. ``"yolo26x.pt"``).  For W&B
        artifacts, download the ``.pt`` locally in the Kaggle notebook first.
    distill_weight:
        Lambda for the feature distillation loss component.
    task_weight:
        Lambda for the detection task loss component.
    """
    from ultralytics import YOLO
    import wandb

    work_dir = work_dir.expanduser().resolve()
    dataset_dir = work_dir / "visdrone_yolo"
    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataset (reuse existing utility)
    train_prepared = prepare_yolo_dataset(
        data_root=data_root, output_root=dataset_dir, split="train"
    )
    val_prepared = prepare_yolo_dataset(
        data_root=data_root, output_root=dataset_dir, split="val"
    )

    # DataLoader
    train_dataset = _VisDroneDataset(
        images_dir=train_prepared.images,
        labels_dir=train_prepared.labels,
        imgsz=imgsz,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_fn,
    )

    # ── Teacher setup ──────────────────────────────────────────────────────
    # Load teacher on CPU first so we can probe layer shapes cheaply.
    print(f"[distill] loading teacher from {teacher_weights!r}")
    teacher = YOLO(teacher_weights)
    teacher_nn: nn.Module = teacher.model
    teacher_nn.eval()
    for p in teacher_nn.parameters():
        p.requires_grad_(False)

    print("[distill] probing teacher FPN layer shapes …")
    fpn_modules, teacher_ch = _find_fpn_layers(teacher_nn, imgsz=imgsz)
    t_channels = (
        teacher_ch.get("P3", _DEFAULT_T_P3),
        teacher_ch.get("P4", _DEFAULT_T_P4),
        teacher_ch.get("P5", _DEFAULT_T_P5),
    )
    print(f"[distill] teacher FPN channels: P3={t_channels[0]}, P4={t_channels[1]}, P5={t_channels[2]}")

    # Determine primary device
    device_ids = [int(d.strip()) for d in device.split(",") if d.strip()]
    use_cuda = torch.cuda.is_available() and len(device_ids) > 0
    primary = torch.device(f"cuda:{device_ids[0]}" if use_cuda else "cpu")

    teacher_nn = teacher_nn.to(primary)
    _register_teacher_hooks(fpn_modules)  # hooks fire on primary GPU after .to()

    # ── Student supernet setup ─────────────────────────────────────────────
    supernet = YOLOSupernet(
        num_classes=len(VISDRONE_NAMES),
        teacher_channels=t_channels,
    )

    last_ckpt = ckpt_dir / "supernet_last.pt"
    start_epoch = 0
    if resume and last_ckpt.exists():
        print(f"[distill] resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location="cpu")
        supernet.load_state_dict(ckpt["supernet_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
    elif pretrained_backbone is not None:
        supernet.load_pretrained_backbone(pretrained_backbone)

    if use_cuda and len(device_ids) > 1:
        supernet = nn.DataParallel(supernet, device_ids=device_ids)
    supernet = supernet.to(primary)

    # ── Optimizer + scheduler ──────────────────────────────────────────────
    inner: YOLOSupernet = (
        supernet.module if isinstance(supernet, nn.DataParallel) else supernet
    )
    optimizer = torch.optim.AdamW(
        supernet.parameters(), lr=lr0, weight_decay=weight_decay
    )
    if resume and last_ckpt.exists():
        optimizer.load_state_dict(ckpt["optimizer_state"])

    warmup_sched = LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1.0,
        total_iters=max(1, warmup_epochs),
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=lr0 * lrf,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )
    # Fast-forward scheduler if resuming
    for _ in range(start_epoch):
        scheduler.step()

    task_loss_fn = _TaskLoss(num_classes=len(VISDRONE_NAMES))

    # ── W&B init ──────────────────────────────────────────────────────────
    git_sha = _git_sha(Path.cwd())
    run_name = run_name or "supernet-distill"

    config = {
        "model": "YOLOSupernet",
        "teacher_weights": teacher_weights,
        "teacher_channels": {"P3": t_channels[0], "P4": t_channels[1], "P5": t_channels[2]},
        "dataset": "VisDrone",
        "kaggle_dataset_slug": "banuprasadb/visdrone-dataset",
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device,
        "lr0": lr0,
        "lrf": lrf,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "distill_weight": distill_weight,
        "task_weight": task_weight,
        "pretrained_backbone": pretrained_backbone,
        "backbone_depth_choices": BACKBONE_DEPTH_CHOICES,
        "neck_depth_choices": NECK_DEPTH_CHOICES,
        "search_space_size": len(BACKBONE_DEPTH_CHOICES) ** 4 * len(NECK_DEPTH_CHOICES) ** 4,
        "accelerator": "NvidiaTeslaT4",
        "expected_gpu_count": len(device_ids),
        "git_sha": git_sha,
        "python": platform.python_version(),
    }

    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        job_type="supernet-distill",
        tags=["visdrone", "supernet", "spos", "kd", "distill", "kaggle"],
        config=config,
        resume="allow" if resume else None,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_loss = float("inf")
    best_ckpt = ckpt_dir / "supernet_best.pt"
    summary_metrics: dict[str, Any] = {}

    try:
        for epoch in range(start_epoch, epochs):
            supernet.train()
            teacher_nn.eval()

            epoch_loss_total = 0.0
            epoch_loss_task = 0.0
            epoch_loss_distill = 0.0
            num_batches = 0
            arch = inner.sample_arch()  # fallback for logging if loader is empty

            for images, targets in train_loader:
                # SPOS: sample a new random sub-architecture every step
                arch = inner.sample_arch()
                inner.set_arch(arch)

                images = images.to(primary, non_blocking=True)
                targets = targets.to(primary, non_blocking=True)
                optimizer.zero_grad()

                # Teacher forward (no grad) — side-effect: fills _TEACHER_FEAT_STORE
                with torch.no_grad():
                    teacher_nn(images)

                # Student forward
                preds, student_feats = supernet(images)

                # Losses
                L_task = task_loss_fn(preds, targets, imgsz)
                L_distill = _distill_loss(student_feats)
                L_total = task_weight * L_task + distill_weight * L_distill

                L_total.backward()
                nn.utils.clip_grad_norm_(supernet.parameters(), max_norm=10.0)
                optimizer.step()

                epoch_loss_total += L_total.item()
                epoch_loss_task += L_task.item()
                epoch_loss_distill += L_distill.item()
                num_batches += 1

            scheduler.step()

            avg_total = epoch_loss_total / max(1, num_batches)
            avg_task = epoch_loss_task / max(1, num_batches)
            avg_distill = epoch_loss_distill / max(1, num_batches)
            current_lr = optimizer.param_groups[0]["lr"]

            log = {
                "epoch": epoch + 1,
                "train/loss_total": avg_total,
                "train/loss_task": avg_task,
                "train/loss_distill": avg_distill,
                "train/lr": current_lr,
                "arch/backbone_depths": str(arch.backbone_depths),
                "arch/neck_depths": str(arch.neck_depths),
            }
            wandb.log(log, step=epoch + 1)
            print(
                f"[epoch {epoch + 1:3d}/{epochs}] "
                f"total={avg_total:.4f} task={avg_task:.4f} "
                f"distill={avg_distill:.4f} lr={current_lr:.2e} "
                f"arch={arch.backbone_depths}+{arch.neck_depths}"
            )

            # Save checkpoint
            state = {
                "epoch": epoch,
                "supernet_state": inner.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "arch_last": arch,
                "loss_total": avg_total,
            }
            torch.save(state, last_ckpt)

            if avg_total < best_loss:
                best_loss = avg_total
                torch.save(state, best_ckpt)

        summary_metrics = {
            "train/best_loss_total": best_loss,
            "dataset/train_image_count": train_prepared.image_count,
            "dataset/val_image_count": val_prepared.image_count,
            "dataset/train_label_count": train_prepared.label_count,
            "run/git_sha": git_sha,
        }
        wandb.log(summary_metrics)
        wandb_run.summary.update(summary_metrics)

        # Upload final supernet as W&B artifact
        if best_ckpt.exists():
            artifact = wandb.Artifact(
                name="yolov26x-supernet-student",
                type="model",
                description=(
                    "SPOS supernet student (YOLOv26s-width, depth-searchable) "
                    "trained with feature KD from YOLOv26x teacher on VisDrone"
                ),
                metadata={**config, **summary_metrics},
            )
            artifact.add_file(str(best_ckpt), name="supernet_best.pt")
            if last_ckpt.exists():
                artifact.add_file(str(last_ckpt), name="supernet_last.pt")
            wandb_run.log_artifact(artifact)

    finally:
        wandb.finish()

    metrics_path = work_dir / "supernet_distill_metrics.json"
    metrics_path.write_text(
        json.dumps(summary_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary_metrics, indent=2, sort_keys=True))
    return summary_metrics


# ── Utilities ──────────────────────────────────────────────────────────────


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA")
