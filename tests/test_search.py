"""Unit tests for architecture search — no GPU, no dataset, no Ultralytics required."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from visdrone_det.search import (
    ArchScore,
    SearchResult,
    evaluate_arch,
    exhaustive_search,
    export_best_student,
    load_best_student,
    random_search,
    run_student_search,
)
from visdrone_det.supernet import ArchConfig, YOLOSupernet


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_model() -> YOLOSupernet:
    """Small test supernet with reduced teacher channels."""
    return YOLOSupernet(num_classes=10, teacher_channels=(64, 128, 128))


def _make_batches(
    n: int = 2,
    imgsz: int = 64,
    batch: int = 1,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Create small fixed-seed random batches for proxy evaluation."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return [torch.randn(batch, 3, imgsz, imgsz, generator=rng) for _ in range(n)]


# ── evaluate_arch ─────────────────────────────────────────────────────────────


def test_evaluate_arch_returns_float():
    model = _make_model()
    model.eval()
    batches = _make_batches()
    arch = ArchConfig(backbone_depths=(1, 1, 1, 1), neck_depths=(1, 1, 1, 1))
    score = evaluate_arch(model, arch, batches, torch.device("cpu"))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_evaluate_arch_empty_batches_returns_zero():
    model = _make_model()
    arch = ArchConfig(backbone_depths=(1, 1, 1, 1), neck_depths=(1, 1, 1, 1))
    score = evaluate_arch(model, arch, [], torch.device("cpu"))
    assert score == 0.0


def test_evaluate_arch_different_depths_produce_valid_scores():
    """Two extreme architectures both produce scores in [0, 1]."""
    model = _make_model()
    model.eval()
    batches = _make_batches(n=2, imgsz=64, seed=99)

    arch_min = ArchConfig(backbone_depths=(1, 1, 1, 1), neck_depths=(1, 1, 1, 1))
    arch_max = ArchConfig(backbone_depths=(3, 3, 3, 3), neck_depths=(2, 2, 2, 2))

    score_min = evaluate_arch(model, arch_min, batches, torch.device("cpu"))
    score_max = evaluate_arch(model, arch_max, batches, torch.device("cpu"))

    assert 0.0 <= score_min <= 1.0
    assert 0.0 <= score_max <= 1.0


# ── random_search ─────────────────────────────────────────────────────────────


def test_random_search_returns_valid_result():
    model = _make_model()
    batches = _make_batches()

    result = random_search(model, batches, n_samples=5, device=torch.device("cpu"), seed=42)

    assert isinstance(result, SearchResult)
    assert len(result.all_scores) == 5
    assert result.n_evaluated == 5
    assert isinstance(result.best_arch, ArchConfig)
    assert result.best_score == result.all_scores[0].score


def test_random_search_scores_sorted_descending():
    model = _make_model()
    batches = _make_batches()

    result = random_search(model, batches, n_samples=8, device=torch.device("cpu"), seed=0)

    scores = [s.score for s in result.all_scores]
    assert scores == sorted(scores, reverse=True)


def test_random_search_rank_field():
    model = _make_model()
    batches = _make_batches()

    result = random_search(model, batches, n_samples=5, device=torch.device("cpu"), seed=1)

    for i, s in enumerate(result.all_scores):
        assert s.rank == i + 1


def test_random_search_best_arch_is_valid():
    model = _make_model()
    batches = _make_batches()

    result = random_search(model, batches, n_samples=6, device=torch.device("cpu"), seed=7)

    from visdrone_det.supernet import BACKBONE_DEPTH_CHOICES, NECK_DEPTH_CHOICES

    assert all(d in BACKBONE_DEPTH_CHOICES for d in result.best_arch.backbone_depths)
    assert all(d in NECK_DEPTH_CHOICES for d in result.best_arch.neck_depths)


def test_random_search_reproducible_with_seed():
    """Same seed → same best architecture."""
    model = _make_model()
    batches = _make_batches()

    r1 = random_search(model, batches, n_samples=5, device=torch.device("cpu"), seed=123)
    r2 = random_search(model, batches, n_samples=5, device=torch.device("cpu"), seed=123)

    assert r1.best_arch.backbone_depths == r2.best_arch.backbone_depths
    assert r1.best_arch.neck_depths == r2.best_arch.neck_depths


# ── exhaustive_search ─────────────────────────────────────────────────────────


def test_exhaustive_search_covers_all(monkeypatch):
    """Exhaustive search must evaluate all 1,296 sub-networks (mocked for speed)."""
    import visdrone_det.search as search_mod

    call_count = [0]

    def mock_evaluate(model, arch, batches, device):
        call_count[0] += 1
        return float(sum(arch.backbone_depths) + sum(arch.neck_depths)) / 20.0

    monkeypatch.setattr(search_mod, "evaluate_arch", mock_evaluate)

    model = _make_model()
    result = exhaustive_search(model, [], torch.device("cpu"))

    assert call_count[0] == 1296  # 3^4 × 2^4
    assert result.n_evaluated == 1296
    assert isinstance(result.best_arch, ArchConfig)
    # With mock, max bd=(3,3,3,3)=[12] nd=(2,2,2,2)=[8] → score=(12+8)/20=1.0
    assert abs(result.best_score - 1.0) < 1e-5


def test_exhaustive_search_scores_sorted(monkeypatch):
    import visdrone_det.search as search_mod

    monkeypatch.setattr(
        search_mod,
        "evaluate_arch",
        lambda model, arch, batches, device: float(sum(arch.backbone_depths)) / 12.0,
    )

    model = _make_model()
    result = exhaustive_search(model, [], torch.device("cpu"))

    scores = [s.score for s in result.all_scores]
    assert scores == sorted(scores, reverse=True)


# ── export_best_student / load_best_student ───────────────────────────────────


def test_export_checkpoint_has_required_keys():
    model = _make_model()
    batches = _make_batches(n=2, imgsz=64)

    result = random_search(model, batches, n_samples=3, device=torch.device("cpu"), seed=7)

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "best_student.pt"
        export_best_student(model, result, save_path)

        assert save_path.exists()
        ckpt = torch.load(save_path, map_location="cpu", weights_only=True)

        for key in ("state_dict", "arch", "teacher_channels", "score", "n_evaluated", "top5"):
            assert key in ckpt, f"missing key: {key}"


def test_export_checkpoint_arch_matches_result():
    model = _make_model()
    batches = _make_batches(n=2, imgsz=64)

    result = random_search(model, batches, n_samples=3, device=torch.device("cpu"), seed=11)

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "student.pt"
        export_best_student(model, result, save_path)

        ckpt = torch.load(save_path, map_location="cpu", weights_only=True)

        assert ckpt["arch"]["backbone_depths"] == list(result.best_arch.backbone_depths)
        assert ckpt["arch"]["neck_depths"] == list(result.best_arch.neck_depths)
        assert abs(ckpt["score"] - result.best_score) < 1e-6
        assert ckpt["n_evaluated"] == result.n_evaluated


def test_export_top5_length():
    model = _make_model()
    batches = _make_batches(n=1, imgsz=64)

    result = random_search(model, batches, n_samples=3, device=torch.device("cpu"), seed=2)

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "student.pt"
        export_best_student(model, result, save_path)

        ckpt = torch.load(save_path, map_location="cpu", weights_only=True)
        # top5 bounded by actual n_samples (3 < 5)
        assert len(ckpt["top5"]) == 3


def test_load_best_student_restores_arch():
    model = _make_model()
    batches = _make_batches(n=1, imgsz=64)

    result = random_search(model, batches, n_samples=3, device=torch.device("cpu"), seed=5)

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "student.pt"
        export_best_student(model, result, save_path)

        loaded_model, loaded_arch = load_best_student(save_path)

        assert isinstance(loaded_model, YOLOSupernet)
        assert loaded_arch.backbone_depths == result.best_arch.backbone_depths
        assert loaded_arch.neck_depths == result.best_arch.neck_depths


def test_load_best_student_weights_match():
    """Loaded model must produce identical outputs to the exported supernet."""
    model = _make_model()
    batches = _make_batches(n=1, imgsz=64)

    result = random_search(model, batches, n_samples=3, device=torch.device("cpu"), seed=9)

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "student.pt"
        export_best_student(model, result, save_path)

        loaded_model, loaded_arch = load_best_student(save_path)

        # Set both models to best arch and eval mode
        model.set_arch(result.best_arch)
        model.eval()
        loaded_model.eval()

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            preds_orig, _ = model(x)
            preds_loaded, _ = loaded_model(x)

        for (cls1, box1), (cls2, box2) in zip(preds_orig, preds_loaded):
            assert torch.allclose(cls1, cls2, atol=1e-5), "cls outputs differ"
            assert torch.allclose(box1, box2, atol=1e-5), "box outputs differ"


# ── run_student_search ────────────────────────────────────────────────────────


def test_run_student_search_dummy_weights_end_to_end():
    """Full pipeline with dummy_weights=True — no real checkpoint needed."""
    with tempfile.TemporaryDirectory() as tmp:
        result = run_student_search(
            supernet_weights=None,
            work_dir=Path(tmp),
            n_samples=3,
            search_mode="random",
            imgsz=64,
            batch=1,
            n_proxy_batches=2,
            device="cpu",
            seed=42,
            dummy_weights=True,
        )

        assert isinstance(result, SearchResult)
        assert result.n_evaluated == 3

        assert (Path(tmp) / "best_student.pt").exists()
        assert (Path(tmp) / "search_results.json").exists()


def test_run_student_search_output_json_structure():
    with tempfile.TemporaryDirectory() as tmp:
        run_student_search(
            supernet_weights=None,
            work_dir=Path(tmp),
            n_samples=4,
            search_mode="random",
            imgsz=64,
            batch=1,
            n_proxy_batches=1,
            device="cpu",
            seed=0,
            dummy_weights=True,
        )

        summary = json.loads((Path(tmp) / "search_results.json").read_text())
        assert "best_arch" in summary
        assert "best_score" in summary
        assert "n_evaluated" in summary
        assert summary["n_evaluated"] == 4
        assert len(summary["top10"]) == 4  # n_samples=4 < 10
        assert "backbone_depths" in summary["best_arch"]
        assert "neck_depths" in summary["best_arch"]


def test_run_student_search_missing_weights_falls_back():
    """A missing checkpoint path should fall back to random init gracefully."""
    with tempfile.TemporaryDirectory() as tmp:
        result = run_student_search(
            supernet_weights="/nonexistent/path/supernet_best.pt",
            work_dir=Path(tmp),
            n_samples=3,
            search_mode="random",
            imgsz=64,
            batch=1,
            n_proxy_batches=1,
            device="cpu",
            seed=0,
        )
        assert isinstance(result, SearchResult)
        assert result.n_evaluated == 3


def test_run_student_search_wrapped_checkpoint(tmp_path):
    """Accepts checkpoints wrapped under 'state_dict' key (distill.py format)."""
    model = _make_model()
    # Save in distill.py-style wrapped format
    ckpt_path = tmp_path / "supernet_best.pt"
    torch.save({"state_dict": model.state_dict(), "epoch": 10}, ckpt_path)

    result = run_student_search(
        supernet_weights=str(ckpt_path),
        work_dir=tmp_path / "out",
        n_samples=3,
        search_mode="random",
        imgsz=64,
        batch=1,
        n_proxy_batches=1,
        device="cpu",
        seed=5,
    )
    assert isinstance(result, SearchResult)


# ── CLI parsing ───────────────────────────────────────────────────────────────


def test_cli_search_student_defaults():
    """search-student subcommand is accepted with all defaults."""
    from visdrone_det.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["search-student"])
    assert args.command == "search-student"
    assert args.n_samples == 50
    assert args.search_mode == "random"
    assert args.imgsz == 320
    assert args.batch == 4
    assert args.n_proxy_batches == 4
    assert args.device == "0"
    assert args.seed == 42
    assert args.dummy_weights is False
    assert args.wandb_project == "distillNas"


def test_cli_search_student_custom_flags():
    from visdrone_det.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "search-student",
        "--supernet-weights", "/path/to/supernet.pt",
        "--n-samples", "100",
        "--search-mode", "exhaustive",
        "--imgsz", "224",
        "--batch", "2",
        "--seed", "7",
        "--dummy-weights",
        "--run-name", "nas-eval",
    ])
    assert args.supernet_weights == "/path/to/supernet.pt"
    assert args.n_samples == 100
    assert args.search_mode == "exhaustive"
    assert args.imgsz == 224
    assert args.batch == 2
    assert args.seed == 7
    assert args.dummy_weights is True
    assert args.run_name == "nas-eval"
