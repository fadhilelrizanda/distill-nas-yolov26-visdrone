"""Supernet distillation with FAMO automatic loss weighting (Liu et al., NeurIPS 2023)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .distill import (
    _TEACHER_FEAT_STORE,
    _TaskLoss,
    _VisDroneDataset,
    _collate_fn,
    _compute_map50,
    _eval_student,
    _find_fpn_layers,
    _git_sha,
    _register_teacher_hooks,
)
from .supernet import (
    BACKBONE_DEPTH_CHOICES,
    NECK_DEPTH_CHOICES,
    YOLOSupernet,
    _DEFAULT_T_P3,
    _DEFAULT_T_P4,
    _DEFAULT_T_P5,
)
from .visdrone import VISDRONE_NAMES, prepare_yolo_dataset


# ── FAMO weight manager ────────────────────────────────────────────────────


class _FAMOWeights:
    """FAMO adaptive loss weights (Liu et al., NeurIPS 2023).

    Maintains n weights that sum to 1. After each optimizer step, a no-grad
    forward pass measures the new losses; weights are updated so that losses
    with smaller decrease receive higher weight next step — equalizing the
    rate of progress across all loss components.
    """

    def __init__(
        self,
        n: int = 4,
        gamma: float = 0.01,
        device: str | torch.device = "cpu",
    ) -> None:
        self.w = torch.ones(n, device=device) / n
        self.gamma = gamma
        self._l_prev: torch.Tensor | None = None

    def weighted_loss(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """Return FAMO-weighted sum; cache current loss values for update."""
        l = torch.stack([x.detach() for x in losses])
        self._l_prev = l
        return sum(self.w[i] * losses[i] for i in range(len(losses)))  # type: ignore[return-value]

    @torch.no_grad()
    def step(self, losses_new: list[torch.Tensor]) -> None:
        """Update weights from loss change measured after the gradient step.

        Losses that decreased less (or increased) get upweighted.
        Losses that decreased quickly get downweighted.
        """
        if self._l_prev is None:
            return
        l_new = torch.stack([x.detach() for x in losses_new])
        delta = l_new - self._l_prev          # positive → loss went up / stagnated
        log_w = torch.log(self.w.clamp(min=1e-8))
        # z_new = z_old + γ·Δl: tasks whose loss decreased (Δl<0) get downweighted
        self.w = torch.softmax(log_w + self.gamma * delta, dim=0)
        self._l_prev = None

    def state_dict(self) -> dict[str, Any]:
        return {"w": self.w.tolist(), "gamma": self.gamma}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.w = torch.tensor(d["w"], device=self.w.device)
        self.gamma = float(d["gamma"])


# ── Loss helpers ───────────────────────────────────────────────────────────


def _compute_losses(
    preds: list[tuple[torch.Tensor, torch.Tensor]],
    student_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targets: torch.Tensor,
    imgsz: int,
    task_loss_fn: _TaskLoss,
) -> list[torch.Tensor]:
    """Return [L_task, L_p3, L_p4, L_p5] — all scalar tensors on student device."""
    L_task = task_loss_fn(preds, targets, imgsz)
    out: list[torch.Tensor] = [L_task]
    for s_feat, key in zip(student_feats, ("P3", "P4", "P5")):
        t_feat = _TEACHER_FEAT_STORE.get(key)
        if t_feat is not None:
            out.append(F.mse_loss(s_feat, t_feat.to(s_feat.device)))
        else:
            out.append(s_feat.new_tensor(0.0))
    return out  # length 4


# ── Main entrypoint ────────────────────────────────────────────────────────


def run_supernet_distill_famo(
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
    famo_gamma: float = 0.01,
    pretrained_backbone: str | None = None,
    cache: bool = False,
    resume: bool = False,
    checkpoint_interval: int = 5,
    wandb_project: str = "distillNas",
    wandb_entity: str | None = None,
    wandb_run_id: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train the YOLO supernet student with FAMO-weighted feature distillation.

    Implements Single-Path One-Shot (SPOS) with FAMO (NeurIPS 2023) to
    automatically balance four loss components per training step:
    L_task (detection BCE+GIoU), L_p3/L_p4/L_p5 (MSE feature distillation).

    FAMO keeps a no-grad re-forward after each gradient step to measure loss
    changes, then updates weights so slower-progressing losses receive more
    attention next step.

    Per-epoch evaluation runs the max-depth sub-architecture on the VisDrone val
    split and logs ``val/map50``, ``val/precision``, ``val/recall`` to W&B.

    Parameters
    ----------
    famo_gamma:
        FAMO weight-update step size. Higher values make weights adapt faster.
        Default 0.01 is conservative; try 0.05–0.1 for faster adaptation.
    checkpoint_interval:
        Upload ``supernet_last.pt`` as a W&B checkpoint artifact every N epochs
        (for mid-run resume).  0 disables periodic uploads.
    wandb_run_id:
        W&B run ID to resume.  If ``--resume`` is used without this flag, the
        run ID is loaded from the checkpoint state dict automatically.
    """
    from ultralytics import YOLO
    import wandb

    work_dir = work_dir.expanduser().resolve()
    dataset_dir = work_dir / "visdrone_yolo"
    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────
    train_prepared = prepare_yolo_dataset(
        data_root=data_root, output_root=dataset_dir, split="train"
    )
    val_prepared = prepare_yolo_dataset(
        data_root=data_root, output_root=dataset_dir, split="val"
    )
    train_loader = DataLoader(
        _VisDroneDataset(
            images_dir=train_prepared.images,
            labels_dir=train_prepared.labels,
            imgsz=imgsz,
        ),
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        _VisDroneDataset(
            images_dir=val_prepared.images,
            labels_dir=val_prepared.labels,
            imgsz=imgsz,
        ),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_fn,
    )

    # ── Teacher ────────────────────────────────────────────────────────────
    print(f"[distill-famo] loading teacher from {teacher_weights!r}")
    teacher = YOLO(teacher_weights)
    teacher_nn: nn.Module = teacher.model
    teacher_nn.eval()
    for p in teacher_nn.parameters():
        p.requires_grad_(False)

    print("[distill-famo] probing teacher FPN layer shapes …")
    fpn_modules, teacher_ch = _find_fpn_layers(teacher_nn, imgsz=imgsz)
    t_channels = (
        teacher_ch.get("P3", _DEFAULT_T_P3),
        teacher_ch.get("P4", _DEFAULT_T_P4),
        teacher_ch.get("P5", _DEFAULT_T_P5),
    )
    print(
        f"[distill-famo] teacher FPN channels: "
        f"P3={t_channels[0]}, P4={t_channels[1]}, P5={t_channels[2]}"
    )

    device_ids = [int(d.strip()) for d in device.split(",") if d.strip()]
    use_cuda = torch.cuda.is_available() and len(device_ids) > 0
    primary = torch.device(f"cuda:{device_ids[0]}" if use_cuda else "cpu")

    teacher_nn = teacher_nn.to(primary)
    _register_teacher_hooks(fpn_modules)

    # ── Student supernet ───────────────────────────────────────────────────
    supernet = YOLOSupernet(
        num_classes=len(VISDRONE_NAMES),
        teacher_channels=t_channels,
    )

    last_ckpt = ckpt_dir / "supernet_last.pt"
    start_epoch = 0
    ckpt: dict[str, Any] = {}
    if resume and last_ckpt.exists():
        print(f"[distill-famo] resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location="cpu")
        supernet.load_state_dict(ckpt["supernet_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        # Recover W&B run ID from checkpoint if not provided on CLI
        if wandb_run_id is None:
            wandb_run_id = ckpt.get("wandb_run_id")
    elif pretrained_backbone is not None:
        supernet.load_pretrained_backbone(pretrained_backbone)

    if use_cuda and len(device_ids) > 1:
        supernet = nn.DataParallel(supernet, device_ids=device_ids)
    supernet = supernet.to(primary)

    inner: YOLOSupernet = (
        supernet.module if isinstance(supernet, nn.DataParallel) else supernet
    )

    # ── Optimizer + scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        supernet.parameters(), lr=lr0, weight_decay=weight_decay
    )
    if resume and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])

    warmup_sched = LinearLR(
        optimizer, start_factor=1e-4, end_factor=1.0, total_iters=max(1, warmup_epochs)
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=lr0 * lrf
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )
    for _ in range(start_epoch):
        scheduler.step()

    # ── FAMO weights ───────────────────────────────────────────────────────
    famo = _FAMOWeights(n=4, gamma=famo_gamma, device=primary)
    if resume and ckpt.get("famo"):
        famo.load_state_dict(ckpt["famo"])
        famo.w = famo.w.to(primary)

    task_loss_fn = _TaskLoss(num_classes=len(VISDRONE_NAMES))

    # ── W&B init ──────────────────────────────────────────────────────────
    git_sha = _git_sha(Path.cwd())
    run_name = run_name or "supernet-distill-famo"

    config = {
        "model": "YOLOSupernet",
        "loss_weighting": "FAMO",
        "famo_gamma": famo_gamma,
        "teacher_weights": teacher_weights,
        "teacher_channels": {
            "P3": t_channels[0], "P4": t_channels[1], "P5": t_channels[2]
        },
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
        "pretrained_backbone": pretrained_backbone,
        "backbone_depth_choices": BACKBONE_DEPTH_CHOICES,
        "neck_depth_choices": NECK_DEPTH_CHOICES,
        "search_space_size": (
            len(BACKBONE_DEPTH_CHOICES) ** 4 * len(NECK_DEPTH_CHOICES) ** 4
        ),
        "accelerator": "NvidiaTeslaT4",
        "expected_gpu_count": len(device_ids),
        "git_sha": git_sha,
        "python": platform.python_version(),
    }

    # Resume an existing W&B run when run_id is known; otherwise start fresh.
    if resume and wandb_run_id:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            id=wandb_run_id,
            resume="must",
            config=config,
        )
    else:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            job_type="supernet-distill-famo",
            tags=["visdrone", "supernet", "spos", "kd", "distill", "famo", "kaggle"],
            config=config,
            resume="allow" if resume else None,
        )
    active_wandb_run_id: str = wandb_run.id

    # ── Training loop ──────────────────────────────────────────────────────
    best_loss  = float("inf")
    best_map50 = ckpt.get("best_map50", 0.0)
    best_ckpt  = ckpt_dir / "supernet_best.pt"
    summary_metrics: dict[str, Any] = {}

    try:
        for epoch in range(start_epoch, epochs):
            supernet.train()
            teacher_nn.eval()

            epoch_loss_total = 0.0
            epoch_loss_task  = 0.0
            epoch_loss_p3    = 0.0
            epoch_loss_p4    = 0.0
            epoch_loss_p5    = 0.0
            num_batches = 0
            arch = inner.sample_arch()  # fallback for logging if loader is empty

            for images, targets in train_loader:
                # SPOS: new random sub-architecture every step
                arch = inner.sample_arch()
                inner.set_arch(arch)

                images  = images.to(primary, non_blocking=True)
                targets = targets.to(primary, non_blocking=True)
                optimizer.zero_grad()

                # ① Teacher forward — populates _TEACHER_FEAT_STORE
                with torch.no_grad():
                    teacher_nn(images)

                # ② Student forward
                preds, student_feats = supernet(images)

                # ③ Compute 4 losses; FAMO produces weighted sum
                losses = _compute_losses(preds, student_feats, targets, imgsz, task_loss_fn)
                _TEACHER_FEAT_STORE.clear()  # free ~171 MB teacher features before backward
                L_total = famo.weighted_loss(losses)

                # ④ Backward + gradient step
                L_total.backward()
                nn.utils.clip_grad_norm_(supernet.parameters(), max_norm=10.0)
                optimizer.step()

                # ⑤ FAMO update — single-sample re-forward (no_grad) to measure loss
                # direction. Batch=1 cuts re-forward peak memory 4× vs full batch.
                with torch.no_grad():
                    imgs1 = images[:1]
                    tgts1 = targets[targets[:, 0] == 0]
                    teacher_nn(imgs1)
                    preds2, feats2 = supernet(imgs1)
                    losses2 = _compute_losses(
                        preds2, feats2, tgts1, imgsz, task_loss_fn
                    )
                    _TEACHER_FEAT_STORE.clear()  # free re-forward teacher features
                famo.step(losses2)

                epoch_loss_total += L_total.item()
                epoch_loss_task  += losses[0].item()
                epoch_loss_p3    += losses[1].item()
                epoch_loss_p4    += losses[2].item()
                epoch_loss_p5    += losses[3].item()
                num_batches += 1

            scheduler.step()
            if use_cuda:
                torch.cuda.empty_cache()  # defragment CUDA allocator pool each epoch

            nb = max(1, num_batches)
            avg_total = epoch_loss_total / nb
            avg_task  = epoch_loss_task  / nb
            avg_p3    = epoch_loss_p3    / nb
            avg_p4    = epoch_loss_p4    / nb
            avg_p5    = epoch_loss_p5    / nb
            current_lr = optimizer.param_groups[0]["lr"]

            # ── Per-epoch val evaluation ───────────────────────────────────
            val_metrics = _eval_student(
                supernet, inner, val_loader, imgsz, primary,
                num_classes=len(VISDRONE_NAMES),
            )
            supernet.train()

            log = {
                "epoch":                  epoch + 1,
                "train/loss_total":       avg_total,
                "train/loss_task":        avg_task,
                "train/loss_p3":          avg_p3,
                "train/loss_p4":          avg_p4,
                "train/loss_p5":          avg_p5,
                "train/lr":               current_lr,
                "famo/w_task":            famo.w[0].item(),
                "famo/w_p3":              famo.w[1].item(),
                "famo/w_p4":              famo.w[2].item(),
                "famo/w_p5":              famo.w[3].item(),
                "arch/backbone_depths":   str(arch.backbone_depths),
                "arch/neck_depths":       str(arch.neck_depths),
                **val_metrics,
            }
            wandb.log(log, step=epoch + 1)
            print(
                f"[epoch {epoch + 1:3d}/{epochs}] "
                f"total={avg_total:.4f} task={avg_task:.4f} "
                f"p3={avg_p3:.4f} p4={avg_p4:.4f} p5={avg_p5:.4f} "
                f"lr={current_lr:.2e} mAP50={val_metrics['val/map50']:.4f} "
                f"famo=[{famo.w[0]:.3f},{famo.w[1]:.3f},{famo.w[2]:.3f},{famo.w[3]:.3f}]"
            )

            state = {
                "epoch":           epoch,
                "supernet_state":  inner.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "famo":            famo.state_dict(),
                "arch_last":       arch,
                "loss_total":      avg_total,
                "best_map50":      best_map50,
                "wandb_run_id":    active_wandb_run_id,
            }
            torch.save(state, last_ckpt)

            # Best checkpoint: prefer higher mAP50; fall back to lower loss
            if val_metrics["val/map50"] > best_map50:
                best_map50 = val_metrics["val/map50"]
                state["best_map50"] = best_map50
                torch.save(state, best_ckpt)
            elif avg_total < best_loss:
                torch.save(state, best_ckpt)
            if avg_total < best_loss:
                best_loss = avg_total

            # Periodic resume artifact upload
            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                resume_art = wandb.Artifact(
                    name=f"{run_name}-resume",
                    type="checkpoint",
                    description=f"Resume checkpoint after epoch {epoch + 1}",
                )
                resume_art.add_file(str(last_ckpt), name="supernet_last.pt")
                wandb_run.log_artifact(resume_art)

        summary_metrics = {
            "train/best_loss_total":    best_loss,
            "val/best_map50":           best_map50,
            "dataset/train_image_count": train_prepared.image_count,
            "dataset/val_image_count":   val_prepared.image_count,
            "dataset/train_label_count": train_prepared.label_count,
            "run/git_sha":              git_sha,
            "famo/final_w_task":        famo.w[0].item(),
            "famo/final_w_p3":          famo.w[1].item(),
            "famo/final_w_p4":          famo.w[2].item(),
            "famo/final_w_p5":          famo.w[3].item(),
        }
        wandb.log(summary_metrics)
        wandb_run.summary.update(summary_metrics)

        if best_ckpt.exists():
            artifact = wandb.Artifact(
                name="yolov26x-supernet-student-famo",
                type="model",
                description=(
                    "SPOS supernet student trained with FAMO-weighted feature KD "
                    "(task + P3/P4/P5 MSE) from YOLOv26x teacher on VisDrone"
                ),
                metadata={**config, **summary_metrics},
            )
            artifact.add_file(str(best_ckpt), name="supernet_best.pt")
            if last_ckpt.exists():
                artifact.add_file(str(last_ckpt), name="supernet_last.pt")
            wandb_run.log_artifact(artifact)

    finally:
        wandb.finish()

    metrics_path = work_dir / "supernet_distill_famo_metrics.json"
    metrics_path.write_text(
        json.dumps(summary_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary_metrics, indent=2, sort_keys=True))
    return summary_metrics
