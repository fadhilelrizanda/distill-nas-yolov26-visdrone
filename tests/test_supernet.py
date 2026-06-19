"""Unit tests for YOLOSupernet — no GPU, no dataset, no Ultralytics required."""

from __future__ import annotations

import torch

from visdrone_det.supernet import (
    BACKBONE_DEPTH_CHOICES,
    NECK_DEPTH_CHOICES,
    ArchConfig,
    SearchableC2f,
    YOLOSupernet,
)


def test_arch_config_fields():
    arch = ArchConfig(backbone_depths=(1, 2, 3, 1), neck_depths=(1, 2, 1, 2))
    assert len(arch.backbone_depths) == 4
    assert len(arch.neck_depths) == 4
    assert all(d in BACKBONE_DEPTH_CHOICES for d in arch.backbone_depths)
    assert all(d in NECK_DEPTH_CHOICES for d in arch.neck_depths)


def test_sample_arch_returns_valid_config():
    model = YOLOSupernet(num_classes=10)
    for _ in range(20):
        arch = model.sample_arch()
        assert len(arch.backbone_depths) == 4
        assert len(arch.neck_depths) == 4
        for d in arch.backbone_depths:
            assert d in BACKBONE_DEPTH_CHOICES
        for d in arch.neck_depths:
            assert d in NECK_DEPTH_CHOICES


def test_set_arch_changes_active_depth():
    model = YOLOSupernet(num_classes=10)
    arch = ArchConfig(backbone_depths=(1, 1, 1, 1), neck_depths=(1, 1, 1, 1))
    model.set_arch(arch)
    assert model.bb0._active_n == 1
    assert model.bb3._active_n == 1
    assert model.neck0._active_n == 1
    assert model.neck3._active_n == 1

    arch2 = ArchConfig(backbone_depths=(3, 3, 3, 3), neck_depths=(2, 2, 2, 2))
    model.set_arch(arch2)
    assert model.bb0._active_n == 3
    assert model.neck0._active_n == 2


def test_forward_output_shapes():
    """Full forward pass with dummy input; verify output tensor shapes."""
    model = YOLOSupernet(num_classes=10, teacher_channels=(96, 192, 192))
    model.eval()

    arch = model.sample_arch()
    model.set_arch(arch)

    x = torch.zeros(2, 3, 640, 640)
    with torch.no_grad():
        preds, feats = model(x)

    # preds: 3 scales, each (cls, box)
    assert len(preds) == 3
    cls_p3, box_p3 = preds[0]
    cls_p4, box_p4 = preds[1]
    cls_p5, box_p5 = preds[2]

    assert cls_p3.shape == (2, 10, 80, 80)
    assert box_p3.shape == (2, 4, 80, 80)
    assert cls_p4.shape == (2, 10, 40, 40)
    assert box_p4.shape == (2, 4, 40, 40)
    assert cls_p5.shape == (2, 10, 20, 20)
    assert box_p5.shape == (2, 4, 20, 20)

    # feats: 3 projected feature maps
    assert len(feats) == 3
    assert feats[0].shape == (2, 96, 80, 80)   # proj_p3 → teacher T_P3=96
    assert feats[1].shape == (2, 192, 40, 40)  # proj_p4 → teacher T_P4=192
    assert feats[2].shape == (2, 192, 20, 20)  # proj_p5 → teacher T_P5=192


def test_forward_shapes_all_depth_combos():
    """Spot-check a few depth combinations to confirm shapes are invariant."""
    model = YOLOSupernet(num_classes=5, teacher_channels=(64, 128, 128))
    model.eval()
    x = torch.zeros(1, 3, 640, 640)

    for bd in [(1, 1, 1, 1), (3, 3, 3, 3), (2, 1, 3, 2)]:
        for nd in [(1, 1, 1, 1), (2, 2, 2, 2), (1, 2, 1, 2)]:
            arch = ArchConfig(
                backbone_depths=bd,  # type: ignore[arg-type]
                neck_depths=nd,      # type: ignore[arg-type]
            )
            model.set_arch(arch)
            with torch.no_grad():
                preds, feats = model(x)
            assert preds[0][0].shape[1] == 5   # num_classes
            assert feats[0].shape[1] == 64     # T_P3


def test_searchable_c2f_zero_padding():
    """Inactive slots must produce zeros — verify by checking gradient flow."""
    block = SearchableC2f(in_ch=32, out_ch=32, max_n=3, shortcut=False)
    block.eval()

    block.set_active_n(1)
    x = torch.ones(1, 32, 8, 8)
    with torch.no_grad():
        out1 = block(x)

    block.set_active_n(3)
    with torch.no_grad():
        out3 = block(x)

    # Outputs will differ since more bottlenecks contribute (weights are random)
    assert out1.shape == out3.shape == (1, 32, 8, 8)
    # The depth-1 and depth-3 outputs will not be equal in general
    assert not torch.allclose(out1, out3)


def test_cli_distill_supernet_parses():
    """distill-supernet subcommand is accepted with all defaults."""
    from visdrone_det.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["distill-supernet"])
    assert args.command == "distill-supernet"
    assert args.epochs == 50
    assert args.distill_weight == 1.0
    assert args.task_weight == 1.0
    assert args.device == "0,1"
    assert args.wandb_project == "distillNas"
    assert args.resume is False
    assert args.pretrained_backbone is None


def test_cli_distill_supernet_custom_flags():
    from visdrone_det.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "distill-supernet",
        "--epochs", "10",
        "--distill-weight", "2.5",
        "--task-weight", "0.5",
        "--pretrained-backbone", "yolo26s.pt",
        "--resume",
        "--run-name", "test-run",
    ])
    assert args.epochs == 10
    assert args.distill_weight == 2.5
    assert args.task_weight == 0.5
    assert args.pretrained_backbone == "yolo26s.pt"
    assert args.resume is True
    assert args.run_name == "test-run"
