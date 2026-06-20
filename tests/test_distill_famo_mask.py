"""Unit tests for _fg_mask and _compute_losses_masked — CPU only, no dataset."""

from __future__ import annotations

import torch

from visdrone_det.distill import _TEACHER_FEAT_STORE, _TaskLoss
from visdrone_det.distill_famo_mask import _compute_losses_masked, _fg_mask


# ── _fg_mask ───────────────────────────────────────────────────────────────


def test_fg_mask_shape():
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])
    mask = _fg_mask(targets, grid_h=8, grid_w=8, batch_size=2, sigma=1.0)
    assert mask.shape == (2, 1, 8, 8)


def test_fg_mask_values_in_range():
    targets = torch.tensor([[0, 0, 0.3, 0.7, 0.1, 0.1], [1, 2, 0.5, 0.5, 0.2, 0.2]])
    mask = _fg_mask(targets, grid_h=10, grid_w=10, batch_size=2, sigma=1.5)
    assert mask.min().item() >= 0.0
    assert mask.max().item() <= 1.0 + 1e-6


def test_fg_mask_no_targets_all_zero():
    targets = torch.zeros(0, 6)
    mask = _fg_mask(targets, grid_h=8, grid_w=8, batch_size=2, sigma=2.0)
    assert mask.sum().item() == 0.0


def test_fg_mask_center_cell_is_max():
    """GT box at (0.5, 0.5) normalized → center cell should have the highest value."""
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])
    gh, gw = 10, 10
    mask = _fg_mask(targets, grid_h=gh, grid_w=gw, batch_size=1, sigma=1.0)
    flat_idx = mask[0, 0].argmax()
    row = (flat_idx // gw).item()
    col = (flat_idx % gw).item()
    # center should be near row=5, col=5 (0.5 * 10 = 5.0)
    assert abs(row - 5) <= 1
    assert abs(col - 5) <= 1


def test_fg_mask_multi_gt_two_peaks():
    """Two GT boxes far apart should produce two distinct high-value regions."""
    targets = torch.tensor([
        [0, 0, 0.1, 0.1, 0.05, 0.05],  # top-left
        [0, 1, 0.9, 0.9, 0.05, 0.05],  # bottom-right
    ])
    gh, gw = 20, 20
    mask = _fg_mask(targets, grid_h=gh, grid_w=gw, batch_size=1, sigma=1.0)
    m = mask[0, 0]
    top_left_val     = m[:5, :5].max().item()
    bottom_right_val = m[15:, 15:].max().item()
    center_val       = m[8:12, 8:12].max().item()
    assert top_left_val > center_val
    assert bottom_right_val > center_val


# ── _compute_losses_masked ─────────────────────────────────────────────────


def test_compute_losses_masked_four_tensors():
    from visdrone_det.supernet import ArchConfig, YOLOSupernet

    model = YOLOSupernet(num_classes=10, teacher_channels=(32, 64, 64))
    arch = ArchConfig(backbone_depths=(1, 1, 1, 1), neck_depths=(1, 1, 1, 1))
    model.set_arch(arch)
    model.eval()

    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        preds, student_feats = model(x)

    _TEACHER_FEAT_STORE["P3"] = torch.zeros(1, 32, 8, 8)
    _TEACHER_FEAT_STORE["P4"] = torch.zeros(1, 64, 4, 4)
    _TEACHER_FEAT_STORE["P5"] = torch.zeros(1, 64, 2, 2)

    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])
    task_fn = _TaskLoss(num_classes=10)
    losses = _compute_losses_masked(
        preds, student_feats, targets, imgsz=64, task_loss_fn=task_fn, mask_sigma=2.0
    )

    assert len(losses) == 4
    for loss in losses:
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


def test_compute_losses_masked_missing_teacher():
    """Missing teacher store → distillation losses default to 0."""
    from visdrone_det.supernet import ArchConfig, YOLOSupernet

    _TEACHER_FEAT_STORE.clear()

    model = YOLOSupernet(num_classes=10, teacher_channels=(32, 64, 64))
    arch = ArchConfig(backbone_depths=(1, 1, 1, 1), neck_depths=(1, 1, 1, 1))
    model.set_arch(arch)
    model.eval()

    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        preds, student_feats = model(x)

    targets = torch.zeros(0, 6)
    task_fn = _TaskLoss(num_classes=10)
    losses = _compute_losses_masked(
        preds, student_feats, targets, imgsz=64, task_loss_fn=task_fn
    )

    assert len(losses) == 4
    assert losses[1].item() == 0.0
    assert losses[2].item() == 0.0
    assert losses[3].item() == 0.0


def test_compute_losses_masked_no_gt_zero_distill():
    """Empty targets → mask is all zero → masked distillation losses are 0."""
    from visdrone_det.supernet import ArchConfig, YOLOSupernet

    model = YOLOSupernet(num_classes=10, teacher_channels=(32, 64, 64))
    arch = ArchConfig(backbone_depths=(1, 1, 1, 1), neck_depths=(1, 1, 1, 1))
    model.set_arch(arch)
    model.eval()

    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        preds, student_feats = model(x)

    _TEACHER_FEAT_STORE["P3"] = torch.ones(1, 32, 8, 8)
    _TEACHER_FEAT_STORE["P4"] = torch.ones(1, 64, 4, 4)
    _TEACHER_FEAT_STORE["P5"] = torch.ones(1, 64, 2, 2)

    targets = torch.zeros(0, 6)   # no GT boxes
    task_fn = _TaskLoss(num_classes=10)
    losses = _compute_losses_masked(
        preds, student_feats, targets, imgsz=64, task_loss_fn=task_fn
    )

    assert losses[1].item() == 0.0
    assert losses[2].item() == 0.0
    assert losses[3].item() == 0.0


# ── CLI ────────────────────────────────────────────────────────────────────


def test_cli_famo_mask_parses():
    from visdrone_det.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["distill-supernet-famo-mask"])
    assert args.command == "distill-supernet-famo-mask"
    assert args.famo_gamma == 0.01
    assert args.mask_sigma == 2.0
    assert args.epochs == 50
    assert args.device == "0,1"
    assert args.wandb_project == "distillNas"
    assert args.resume is False


def test_cli_famo_mask_custom_flags():
    from visdrone_det.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "distill-supernet-famo-mask",
        "--famo-gamma", "0.05",
        "--mask-sigma", "3.0",
        "--epochs", "10",
        "--pretrained-backbone", "yolo26s.pt",
        "--resume",
        "--run-name", "test-famo-mask",
    ])
    assert args.famo_gamma == 0.05
    assert args.mask_sigma == 3.0
    assert args.epochs == 10
    assert args.pretrained_backbone == "yolo26s.pt"
    assert args.resume is True
    assert args.run_name == "test-famo-mask"


def test_legacy_commands_still_parse():
    """Both legacy commands must remain unaffected."""
    from visdrone_det.cli import build_parser

    parser = build_parser()

    args = parser.parse_args(["distill-supernet", "--distill-weight", "2.0"])
    assert args.distill_weight == 2.0
    assert args.task_weight == 1.0

    args = parser.parse_args(["distill-supernet-famo", "--famo-gamma", "0.05"])
    assert args.famo_gamma == 0.05
    assert not hasattr(args, "mask_sigma")
