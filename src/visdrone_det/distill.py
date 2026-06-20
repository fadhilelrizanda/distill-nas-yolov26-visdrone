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


# ── Validation helpers ─────────────────────────────────────────────────────

_EVAL_STRIDES = [8, 16, 32]
_EVAL_CONF_THR = 0.01   # low threshold — NMS cleans up false positives
_EVAL_NMS_IOU  = 0.45


@torch.no_grad()
def _decode_preds_single_scale(
    cls_logits: torch.Tensor,   # [B, nc, gh, gw]
    box_deltas: torch.Tensor,   # [B, 4, gh, gw]
    stride: int,
    conf_threshold: float = _EVAL_CONF_THR,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Decode raw head outputs for one FPN scale into per-image detections.

    Returns a list of length B, each element is a tuple
    ``(boxes [N,4] xyxy pixels, scores [N], class_ids [N])``.
    An image with no detections above the threshold → empty tensors.
    """
    B, nc, gh, gw = cls_logits.shape
    device = cls_logits.device
    results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    # Pre-build grid meshes [gh, gw]
    cols = torch.arange(gw, device=device, dtype=torch.float32)
    rows = torch.arange(gh, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")  # [gh, gw]

    # Decode all boxes at once: [B, 4, gh, gw]
    cx = (grid_x.unsqueeze(0) + torch.sigmoid(box_deltas[:, 0])) * stride
    cy = (grid_y.unsqueeze(0) + torch.sigmoid(box_deltas[:, 1])) * stride
    w  = torch.exp(box_deltas[:, 2].clamp(max=4.0)) * stride
    h  = torch.exp(box_deltas[:, 3].clamp(max=4.0)) * stride
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Class scores: [B, nc, gh, gw] → sigmoid
    scores_all = torch.sigmoid(cls_logits)
    max_scores, class_ids = scores_all.max(dim=1)  # [B, gh, gw]

    for b in range(B):
        mask = max_scores[b] > conf_threshold   # [gh, gw] bool
        if not mask.any():
            empty = torch.zeros((0,), device=device)
            results.append((
                torch.zeros((0, 4), device=device),
                empty,
                empty,
            ))
            continue

        boxes = torch.stack([
            x1[b][mask], y1[b][mask],
            x2[b][mask], y2[b][mask],
        ], dim=1)  # [N, 4]
        scores = max_scores[b][mask]        # [N]
        cids   = class_ids[b][mask].float() # [N]
        results.append((boxes, scores, cids))

    return results


@torch.no_grad()
def _eval_student(
    supernet: nn.Module,
    inner: "YOLOSupernet",
    val_loader: DataLoader,
    imgsz: int,
    primary: torch.device,
    num_classes: int = 10,
) -> dict[str, float]:
    """Evaluate the student supernet on the val split.

    Uses the max-depth sub-architecture for a stable, reproducible eval across
    epochs (no randomness from SPOS sampling).  NMS is applied per image across
    all three FPN scales.

    Returns a dict with keys ``val/map50``, ``val/precision``, ``val/recall``.
    """
    try:
        from torchvision.ops import nms as tv_nms, box_iou as tv_box_iou
    except ImportError:
        # torchvision not available — return zeros so training continues
        print("[eval] torchvision not available; skipping val eval", flush=True)
        return {"val/map50": 0.0, "val/precision": 0.0, "val/recall": 0.0}

    # Fix a deterministic max-depth arch for eval comparability
    eval_arch = ArchConfig(
        backbone_depths=(3, 3, 3, 3),
        neck_depths=(2, 2, 2, 2),
    )
    inner.set_arch(eval_arch)
    supernet.eval()

    # all_preds[i] = list of (boxes [N,4], scores [N], class_ids [N]) per image
    # all_gts[i]   = list of (boxes [M,4], class_ids [M]) per image
    all_preds: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    all_gts:   list[tuple[torch.Tensor, torch.Tensor]] = []

    for images, targets in val_loader:
        images  = images.to(primary, non_blocking=True)
        targets = targets.to(primary, non_blocking=True)
        B = images.shape[0]

        preds, _ = supernet(images)

        # Decode predictions per scale
        scale_preds: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        for (cls_logits, box_deltas), stride in zip(preds, _EVAL_STRIDES):
            scale_preds.append(
                _decode_preds_single_scale(cls_logits, box_deltas, stride)
            )

        # Merge scales and apply NMS per image
        for b in range(B):
            boxes_list:  list[torch.Tensor] = []
            scores_list: list[torch.Tensor] = []
            cids_list:   list[torch.Tensor] = []
            for sp in scale_preds:
                bx, sc, ci = sp[b]
                if bx.shape[0] > 0:
                    boxes_list.append(bx)
                    scores_list.append(sc)
                    cids_list.append(ci)

            if boxes_list:
                all_boxes  = torch.cat(boxes_list,  dim=0)  # [K, 4]
                all_scores = torch.cat(scores_list, dim=0)  # [K]
                all_cids   = torch.cat(cids_list,   dim=0)  # [K]
                keep = tv_nms(all_boxes, all_scores, _EVAL_NMS_IOU)
                all_preds.append((all_boxes[keep], all_scores[keep], all_cids[keep]))
            else:
                empty = torch.zeros((0,), device=primary)
                all_preds.append((torch.zeros((0, 4), device=primary), empty, empty))

            # GT boxes for this image: targets rows with batch_idx == b
            row_mask = targets[:, 0] == b
            gt_rows = targets[row_mask]  # [M, 6]: (bi, cls, cx_n, cy_n, w_n, h_n)
            if gt_rows.shape[0] > 0:
                cx_n = gt_rows[:, 2]
                cy_n = gt_rows[:, 3]
                w_n  = gt_rows[:, 4]
                h_n  = gt_rows[:, 5]
                gt_x1 = (cx_n - w_n / 2) * imgsz
                gt_y1 = (cy_n - h_n / 2) * imgsz
                gt_x2 = (cx_n + w_n / 2) * imgsz
                gt_y2 = (cy_n + h_n / 2) * imgsz
                gt_boxes = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=1)
                gt_cids  = gt_rows[:, 1]
                all_gts.append((gt_boxes, gt_cids))
            else:
                empty = torch.zeros((0,), device=primary)
                all_gts.append((torch.zeros((0, 4), device=primary), empty))

    return _compute_map50(all_preds, all_gts, num_classes, iou_thr=0.5)


def _compute_map50(
    all_preds: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    all_gts:   list[tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    iou_thr: float = 0.5,
) -> dict[str, float]:
    """Compute mAP50, mean precision, and mean recall across classes.

    Parameters
    ----------
    all_preds:
        Per-image list of ``(boxes [N,4] xyxy, scores [N], class_ids [N])``.
    all_gts:
        Per-image list of ``(boxes [M,4] xyxy, class_ids [M])``.
    num_classes:
        Number of detection classes.
    iou_thr:
        IoU threshold for a true positive (default 0.5 → mAP50).

    Returns
    -------
    dict with keys ``val/map50``, ``val/precision``, ``val/recall``.
    """
    try:
        from torchvision.ops import box_iou as tv_box_iou
    except ImportError:
        return {"val/map50": 0.0, "val/precision": 0.0, "val/recall": 0.0}

    # Per-class accumulation: list of (confidence, is_tp) tuples
    cls_tp:  dict[int, list[tuple[float, int]]] = {c: [] for c in range(num_classes)}
    cls_ngt: dict[int, int]                      = {c: 0  for c in range(num_classes)}

    for (pred_boxes, pred_scores, pred_cids), (gt_boxes, gt_cids) in zip(all_preds, all_gts):
        # Count GT per class
        for c in gt_cids.long().tolist():
            if 0 <= c < num_classes:
                cls_ngt[c] += 1

        if pred_boxes.shape[0] == 0:
            continue

        # Sort predictions by confidence descending
        order = pred_scores.argsort(descending=True)
        pred_boxes  = pred_boxes[order]
        pred_scores = pred_scores[order]
        pred_cids   = pred_cids[order]

        if gt_boxes.shape[0] == 0:
            # All predictions are false positives
            for i in range(pred_boxes.shape[0]):
                c = int(pred_cids[i].item())
                if 0 <= c < num_classes:
                    cls_tp[c].append((pred_scores[i].item(), 0))
            continue

        # Match predictions to GT greedily per class
        matched_gt = set()
        ious = tv_box_iou(pred_boxes, gt_boxes)  # [Np, Ng]

        for i in range(pred_boxes.shape[0]):
            c = int(pred_cids[i].item())
            if not (0 <= c < num_classes):
                continue
            conf = pred_scores[i].item()

            # Find same-class GTs
            same_class_mask = (gt_cids.long() == c)
            same_class_idx  = same_class_mask.nonzero(as_tuple=False).squeeze(1)

            is_tp = 0
            if same_class_idx.numel() > 0:
                row_ious = ious[i, same_class_idx]  # [M_c]
                best_val, best_pos = row_ious.max(dim=0)
                best_gt_idx = same_class_idx[best_pos].item()
                if best_val.item() >= iou_thr and best_gt_idx not in matched_gt:
                    is_tp = 1
                    matched_gt.add(best_gt_idx)

            cls_tp[c].append((conf, is_tp))

    # Compute AP per class
    aps:       list[float] = []
    precs:     list[float] = []
    recs:      list[float] = []

    for c in range(num_classes):
        ngt = cls_ngt[c]
        det = cls_tp[c]
        if ngt == 0 and len(det) == 0:
            continue
        if ngt == 0:
            aps.append(0.0)
            precs.append(0.0)
            recs.append(0.0)
            continue

        det.sort(key=lambda x: -x[0])  # sort by confidence desc
        tp_cumsum = 0
        fp_cumsum = 0
        prec_pts:  list[float] = []
        rec_pts:   list[float] = []

        for _, is_tp in det:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            prec_pts.append(tp_cumsum / (tp_cumsum + fp_cumsum))
            rec_pts.append(tp_cumsum / ngt)

        # AP via trapezoidal rule
        ap = 0.0
        for k in range(1, len(prec_pts)):
            ap += (rec_pts[k] - rec_pts[k - 1]) * prec_pts[k]
        aps.append(ap)

        # Final precision/recall at the last detection point
        precs.append(prec_pts[-1] if prec_pts else 0.0)
        recs.append(rec_pts[-1]  if rec_pts  else 0.0)

    map50     = float(np.mean(aps))     if aps   else 0.0
    precision = float(np.mean(precs))   if precs else 0.0
    recall    = float(np.mean(recs))    if recs  else 0.0
    return {"val/map50": map50, "val/precision": precision, "val/recall": recall}


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
    checkpoint_interval: int = 5,
    wandb_project: str = "distillNas",
    wandb_entity: str | None = None,
    wandb_run_id: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train the YOLO supernet student with feature distillation from a teacher.

    Implements Single-Path One-Shot (SPOS): one random sub-architecture is
    sampled and activated per step (mini-batch).  Total loss =
    task_weight * L_task + distill_weight * L_distill, where L_distill is MSE
    between student and teacher FPN features at P3/P4/P5 scales.

    Per-epoch evaluation runs the max-depth sub-architecture on the VisDrone val
    split and logs ``val/map50``, ``val/precision``, ``val/recall`` to W&B.

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

    # Prepare dataset (reuse existing utility)
    train_prepared = prepare_yolo_dataset(
        data_root=data_root, output_root=dataset_dir, split="train"
    )
    val_prepared = prepare_yolo_dataset(
        data_root=data_root, output_root=dataset_dir, split="val"
    )

    # DataLoaders
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
    ckpt: dict[str, Any] = {}
    if resume and last_ckpt.exists():
        print(f"[distill] resuming from {last_ckpt}")
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

    # ── Optimizer + scheduler ──────────────────────────────────────────────
    inner: YOLOSupernet = (
        supernet.module if isinstance(supernet, nn.DataParallel) else supernet
    )
    optimizer = torch.optim.AdamW(
        supernet.parameters(), lr=lr0, weight_decay=weight_decay
    )
    if resume and ckpt.get("optimizer_state"):
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
            job_type="supernet-distill",
            tags=["visdrone", "supernet", "spos", "kd", "distill", "kaggle"],
            config=config,
            resume="allow" if resume else None,
        )
    # Always store the live run ID so it propagates into checkpoints
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
                _TEACHER_FEAT_STORE.clear()  # free ~171 MB of teacher features before backward
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

            # ── Per-epoch val evaluation ───────────────────────────────────
            val_metrics = _eval_student(
                supernet, inner, val_loader, imgsz, primary,
                num_classes=len(VISDRONE_NAMES),
            )
            # Restore train mode after eval
            supernet.train()

            log = {
                "epoch": epoch + 1,
                "train/loss_total": avg_total,
                "train/loss_task": avg_task,
                "train/loss_distill": avg_distill,
                "train/lr": current_lr,
                "arch/backbone_depths": str(arch.backbone_depths),
                "arch/neck_depths": str(arch.neck_depths),
                **val_metrics,
            }
            wandb.log(log, step=epoch + 1)
            print(
                f"[epoch {epoch + 1:3d}/{epochs}] "
                f"total={avg_total:.4f} task={avg_task:.4f} "
                f"distill={avg_distill:.4f} lr={current_lr:.2e} "
                f"mAP50={val_metrics['val/map50']:.4f} "
                f"arch={arch.backbone_depths}+{arch.neck_depths}"
            )

            # Save checkpoint (includes W&B run ID for resume continuity)
            state = {
                "epoch": epoch,
                "supernet_state": inner.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "arch_last": arch,
                "loss_total": avg_total,
                "best_map50": best_map50,
                "wandb_run_id": active_wandb_run_id,
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

            if use_cuda:
                torch.cuda.empty_cache()  # defragment CUDA allocator pool each epoch

        summary_metrics = {
            "train/best_loss_total": best_loss,
            "val/best_map50": best_map50,
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
