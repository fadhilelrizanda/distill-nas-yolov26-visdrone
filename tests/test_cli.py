import ast
import os
import tempfile
from pathlib import Path

from visdrone_det.cli import build_parser
from visdrone_det.patches import patch_generate_ddp_file


def test_benchmark_cli_parses_defaults():
    parser = build_parser()
    args = parser.parse_args(["benchmark-yolov26x"])

    assert args.command == "benchmark-yolov26x"
    assert args.model == "yolo26x.pt"
    assert args.wandb_project == "distillNas"
    assert args.device == "0,1"


def test_finetune_cli_parses_live_batch_log():
    """--live-batch-log flag is accepted and defaults to False."""
    parser = build_parser()
    args = parser.parse_args(["finetune-yolov26x"])
    assert args.live_batch_log is False

    args = parser.parse_args(["finetune-yolov26x", "--live-batch-log"])
    assert args.live_batch_log is True


def test_injection_template_produces_valid_python():
    """Verify that the patched DDP file generator writes valid Python."""
    import ultralytics.utils.dist as dist_module

    _original = dist_module.generate_ddp_file
    try:
        patch_generate_ddp_file(checkpoint_interval=1, live_batch_log=False)

        class _MockArgs:
            model = "yolo26x.pt"
            augmentations = None
            resume = False

        class _MockTrainer:
            args = _MockArgs()
            hub_session = None
            save_dir = Path(tempfile.mkdtemp())
            resume = False
            world_size = 2

        filepath = dist_module.generate_ddp_file(_MockTrainer())
        content = Path(filepath).read_text()

        # Verify injection markers
        assert "Custom DDP metrics logging" in content
        assert "on_fit_epoch_end" in content
        assert "on_model_save" in content
        # live_batch_log=False → template substitutes "if False:"
        assert "if False:" in content
        # checkpoint_interval=1 → substituted directly into condition
        assert "epoch % 1 != 0" in content

        # Must be syntactically valid Python
        ast.parse(content)
    finally:
        dist_module.generate_ddp_file = _original
        if "filepath" in locals():
            os.remove(filepath)
