"""Unit tests for _FAMOWeights and _compute_losses — CPU only, no dataset."""

from __future__ import annotations

import torch

from visdrone_det.distill import _TEACHER_FEAT_STORE, _TaskLoss
from visdrone_det.distill_famo import _FAMOWeights, _compute_losses


def test_famo_initial_weights_uniform():
    famo = _FAMOWeights(n=4, gamma=0.01)
    assert famo.w.shape == (4,)
    assert abs(famo.w.sum().item() - 1.0) < 1e-6
    assert torch.allclose(famo.w, torch.full((4,), 0.25))


def test_famo_weights_sum_to_one_after_step():
    famo = _FAMOWeights(n=4, gamma=0.01)
    losses = [torch.tensor(x) for x in [1.0, 0.5, 0.3, 0.2]]
    famo.cache_prev(losses)
    losses2 = [torch.tensor(x) for x in [0.9, 0.49, 0.31, 0.19]]
    famo.step(losses2)
    assert abs(famo.w.sum().item() - 1.0) < 1e-5


def test_famo_stagnant_loss_gets_higher_weight():
    """A loss that does not decrease should receive a higher weight after update."""
    famo = _FAMOWeights(n=2, gamma=1.0)
    famo.cache_prev([torch.tensor(1.0), torch.tensor(1.0)])
    # loss[0] drops a lot; loss[1] stays flat
    famo.step([torch.tensor(0.1), torch.tensor(1.0)])
    assert famo.w[1].item() > famo.w[0].item()


def test_famo_weights_positive():
    famo = _FAMOWeights(n=4, gamma=0.05)
    losses = [torch.tensor(float(i + 1)) for i in range(4)]
    famo.cache_prev(losses)
    losses2 = [torch.tensor(float(i + 1) * 0.9) for i in range(4)]
    famo.step(losses2)
    assert (famo.w > 0).all()


def test_famo_state_dict_roundtrip():
    famo = _FAMOWeights(n=4, gamma=0.05)
    losses = [torch.tensor(float(i + 1)) for i in range(4)]
    famo.cache_prev(losses)
    famo.step([torch.tensor(float(i + 1) * 0.9) for i in range(4)])
    sd = famo.state_dict()

    famo2 = _FAMOWeights(n=4, gamma=0.01)
    famo2.load_state_dict(sd)
    assert torch.allclose(famo.w, famo2.w)
    assert famo2.gamma == 0.05


def test_famo_step_clears_prev():
    famo = _FAMOWeights(n=2, gamma=0.01)
    famo.cache_prev([torch.tensor(1.0), torch.tensor(1.0)])
    assert famo._l_prev is not None
    famo.step([torch.tensor(0.9), torch.tensor(0.95)])
    assert famo._l_prev is None


def test_famo_w_task_never_below_floor():
    """w_task (index 0) must stay >= 0.10 even when task loss drops fast."""
    famo = _FAMOWeights(n=4, gamma=10.0)  # huge gamma to force collapse
    # task loss drops massively; distill losses stay flat
    famo.cache_prev([torch.tensor(10.0), torch.tensor(0.1), torch.tensor(0.1), torch.tensor(0.1)])
    famo.step([torch.tensor(0.001), torch.tensor(0.1), torch.tensor(0.1), torch.tensor(0.1)])
    assert famo.w[0].item() >= 0.10 - 1e-6
    assert abs(famo.w.sum().item() - 1.0) < 1e-5


def test_compute_losses_returns_four_tensors():
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

    targets = torch.zeros(0, 6)
    task_fn = _TaskLoss(num_classes=10)
    losses = _compute_losses(preds, student_feats, targets, imgsz=64, task_loss_fn=task_fn)

    assert len(losses) == 4
    for loss in losses:
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


def test_compute_losses_missing_teacher_store():
    """If teacher store is empty, distill losses should be zero tensors."""
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
    losses = _compute_losses(preds, student_feats, targets, imgsz=64, task_loss_fn=task_fn)

    assert len(losses) == 4
    # distill losses default to 0 when teacher store is empty
    assert losses[1].item() == 0.0
    assert losses[2].item() == 0.0
    assert losses[3].item() == 0.0


def test_cli_distill_supernet_famo_parses():
    from visdrone_det.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["distill-supernet-famo"])
    assert args.command == "distill-supernet-famo"
    assert args.famo_gamma == 0.01
    assert args.epochs == 50
    assert args.device == "0,1"
    assert args.wandb_project == "distillNas"
    assert args.resume is False
    assert args.pretrained_backbone is None


def test_cli_distill_supernet_famo_custom_flags():
    from visdrone_det.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "distill-supernet-famo",
        "--famo-gamma", "0.05",
        "--epochs", "10",
        "--pretrained-backbone", "yolo26s.pt",
        "--resume",
        "--run-name", "test-famo",
    ])
    assert args.famo_gamma == 0.05
    assert args.epochs == 10
    assert args.pretrained_backbone == "yolo26s.pt"
    assert args.resume is True
    assert args.run_name == "test-famo"


def test_legacy_distill_supernet_still_parses():
    """Existing CLI command must be untouched."""
    from visdrone_det.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["distill-supernet", "--distill-weight", "2.0"])
    assert args.distill_weight == 2.0
    assert args.task_weight == 1.0
