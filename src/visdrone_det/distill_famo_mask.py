"""Supernet distillation: FAMO loss weighting + foreground-masked MSE distillation."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import torch

from .distill import _TEACHER_FEAT_STORE, _TaskLoss
from .distill_famo import _run_famo_training_loop


# ── Foreground mask ────────────────────────────────────────────────────────


@torch.no_grad()
def _fg_mask(
    targets: torch.Tensor,
    grid_h: int,
    grid_w: int,
    batch_size: int,
    sigma: float = 2.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a Gaussian foreground heatmap from GT box centers.

    For each GT box the heatmap peaks at 1.0 at the center grid cell and
    falls off as a 2-D Gaussian with the given sigma (in grid cells).
    Multiple GT boxes are combined by pixel-wise max.

    Returns [B, 1, Gh, Gw] float32 with values in [0, 1].
    Empty targets → all-zero mask → distillation loss zeroed out for that image.
    """
    mask = torch.zeros(batch_size, 1, grid_h, grid_w, device=device)
    if targets.numel() == 0:
        return mask

    gy = torch.arange(grid_h, device=device, dtype=torch.float32).unsqueeze(1)
    gx = torch.arange(grid_w, device=device, dtype=torch.float32).unsqueeze(0)

    denom = 2.0 * sigma * sigma
    for row in targets:
        b    = int(row[0].item())
        cx_n = row[2].item()
        cy_n = row[3].item()
        ci   = cy_n * grid_h
        cj   = cx_n * grid_w
        heat = torch.exp(-((gy - ci) ** 2 + (gx - cj) ** 2) / denom)
        mask[b, 0] = torch.max(mask[b, 0], heat)

    return mask


# ── Loss helpers ───────────────────────────────────────────────────────────


def _compute_losses_masked(
    preds: list[tuple[torch.Tensor, torch.Tensor]],
    student_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targets: torch.Tensor,
    imgsz: int,
    task_loss_fn: _TaskLoss,
    mask_sigma: float = 2.0,
) -> list[torch.Tensor]:
    """Return [L_task, L_p3, L_p4, L_p5] with foreground-masked distillation MSE.

    Each distillation loss is MSE weighted by a Gaussian heatmap built from
    GT box centers, so background cells contribute almost nothing.
    """
    L_task = task_loss_fn(preds, targets, imgsz)
    out: list[torch.Tensor] = [L_task]
    B = student_feats[0].shape[0]
    strides = [8, 16, 32]
    for s_feat, key, stride in zip(student_feats, ("P3", "P4", "P5"), strides):
        t_feat = _TEACHER_FEAT_STORE.get(key)
        if t_feat is None:
            out.append(s_feat.new_tensor(0.0))
            continue
        gh, gw = imgsz // stride, imgsz // stride
        mask  = _fg_mask(targets, gh, gw, B, sigma=mask_sigma, device=s_feat.device)
        diff2 = (s_feat - t_feat.to(s_feat.device)) ** 2
        C     = s_feat.shape[1]
        out.append((diff2 * mask).sum() / (mask.sum().clamp(min=1.0) * C))
    return out


# ── Main entrypoint ────────────────────────────────────────────────────────


def run_supernet_distill_famo_mask(
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
    pretrain_epochs: int = 0,
    famo_gamma: float = 0.01,
    mask_sigma: float = 2.0,
    pretrained_backbone: str | None = None,
    cache: bool = False,
    resume: bool = False,
    checkpoint_interval: int = 5,
    wandb_project: str = "distillNas",
    wandb_entity: str | None = None,
    wandb_run_id: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train the YOLO supernet with FAMO weighting + foreground-masked distillation."""
    return _run_famo_training_loop(
        data_root=data_root, work_dir=work_dir, teacher_weights=teacher_weights,
        epochs=epochs, imgsz=imgsz, batch=batch, workers=workers, device=device,
        lr0=lr0, lrf=lrf, weight_decay=weight_decay, warmup_epochs=warmup_epochs,
        pretrain_epochs=pretrain_epochs,
        famo_gamma=famo_gamma, pretrained_backbone=pretrained_backbone, cache=cache,
        resume=resume, checkpoint_interval=checkpoint_interval,
        wandb_project=wandb_project, wandb_entity=wandb_entity,
        wandb_run_id=wandb_run_id, run_name=run_name,
        compute_losses_fn=functools.partial(_compute_losses_masked, mask_sigma=mask_sigma),
        teacher_half=False,
        run_label="famo-mask",
        artifact_name="yolov26x-supernet-student-famo-mask",
        artifact_desc=(
            "SPOS supernet student trained with FAMO-weighted foreground-masked "
            "feature KD (task + masked P3/P4/P5 MSE) from YOLOv26x teacher on VisDrone"
        ),
        extra_config={"distill_masking": "foreground_gaussian", "mask_sigma": mask_sigma},
        extra_epoch0_log={"distill/mask_sigma": mask_sigma},
        extra_tags=["fg-mask"],
        metrics_filename="supernet_distill_famo_mask_metrics.json",
    )
