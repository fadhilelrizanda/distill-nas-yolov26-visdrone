"""Architecture search over the trained SPOS supernet to find the best student sub-network."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .supernet import ArchConfig, BACKBONE_DEPTH_CHOICES, NECK_DEPTH_CHOICES, YOLOSupernet


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class ArchScore:
    """Score for a single evaluated sub-architecture."""

    arch: ArchConfig
    score: float  # higher is better
    rank: int = 0


@dataclass
class SearchResult:
    """Output of one architecture search run."""

    best_arch: ArchConfig
    best_score: float
    all_scores: list[ArchScore]
    n_evaluated: int


# ── Proxy metric ──────────────────────────────────────────────────────────────


def _proxy_score(
    model: YOLOSupernet,
    batch: torch.Tensor,
    device: torch.device,
) -> float:
    """Mean max-confidence across all detection scales for a single batch.

    Runs the currently-active sub-architecture without labels, returning the
    average of max sigmoid(cls_logit) over P3/P4/P5 spatial positions.  Higher
    means the model finds more confident activations for this architecture.

    This is a fast, dataset-agnostic proxy.  On Kaggle with real data, replace
    with a proper mAP50 computation.
    """
    model.eval()
    x = batch.to(device)
    with torch.no_grad():
        preds, _ = model(x)

    scale_scores: list[float] = []
    for cls_logits, _ in preds:
        # cls_logits: [B, nc, H, W] → max confidence across classes+space
        confidence = torch.sigmoid(cls_logits)          # [B, nc, H, W]
        max_conf = confidence.amax(dim=(1, 2, 3))       # [B]
        scale_scores.append(max_conf.mean().item())

    return float(sum(scale_scores) / len(scale_scores))


# ── Evaluation helpers ────────────────────────────────────────────────────────


def evaluate_arch(
    model: YOLOSupernet,
    arch: ArchConfig,
    val_batches: list[torch.Tensor],
    device: torch.device,
) -> float:
    """Activate *arch* and average the proxy score over *val_batches*."""
    model.set_arch(arch)
    if not val_batches:
        return 0.0
    scores = [_proxy_score(model, b, device) for b in val_batches]
    return float(sum(scores) / len(scores))


# ── Search strategies ─────────────────────────────────────────────────────────


def random_search(
    model: YOLOSupernet,
    val_batches: list[torch.Tensor],
    n_samples: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> SearchResult:
    """Sample *n_samples* random sub-architectures and return the ranked results.

    Parameters
    ----------
    model:
        Trained (or dummy) supernet.
    val_batches:
        Fixed list of image tensors ``[B, 3, H, W]`` used as proxy validation.
        Use the same batches for all architectures so scores are comparable.
    n_samples:
        Number of architectures to sample from the 1,296-sub-network space.
    device:
        CPU or CUDA device for inference.
    seed:
        Optional random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    model = model.to(device)
    all_scores: list[ArchScore] = []

    for i in range(n_samples):
        arch = model.sample_arch()
        score = evaluate_arch(model, arch, val_batches, device)
        all_scores.append(ArchScore(arch=arch, score=score))
        print(
            f"  [{i + 1}/{n_samples}] "
            f"bd={arch.backbone_depths} nd={arch.neck_depths} "
            f"→ proxy={score:.4f}"
        )

    all_scores.sort(key=lambda s: s.score, reverse=True)
    for rank, s in enumerate(all_scores):
        s.rank = rank + 1

    return SearchResult(
        best_arch=all_scores[0].arch,
        best_score=all_scores[0].score,
        all_scores=all_scores,
        n_evaluated=n_samples,
    )


def exhaustive_search(
    model: YOLOSupernet,
    val_batches: list[torch.Tensor],
    device: torch.device,
) -> SearchResult:
    """Evaluate all 3^4 × 2^4 = 1,296 sub-architectures and return the best.

    Warning: evaluates every possible sub-network in the search space.
    Use only on Kaggle or with a tiny proxy dataset.  For local testing,
    use ``random_search`` with a small ``n_samples``.
    """
    from itertools import product

    all_backbone = list(product(BACKBONE_DEPTH_CHOICES, repeat=4))
    all_neck = list(product(NECK_DEPTH_CHOICES, repeat=4))
    total = len(all_backbone) * len(all_neck)

    model = model.to(device)
    all_scores: list[ArchScore] = []

    i = 0
    for bd in all_backbone:
        for nd in all_neck:
            arch = ArchConfig(backbone_depths=bd, neck_depths=nd)  # type: ignore[arg-type]
            score = evaluate_arch(model, arch, val_batches, device)
            all_scores.append(ArchScore(arch=arch, score=score))
            if (i + 1) % 100 == 0 or i == 0:
                print(
                    f"  [{i + 1}/{total}] "
                    f"bd={bd} nd={nd} "
                    f"→ proxy={score:.4f}"
                )
            i += 1

    all_scores.sort(key=lambda s: s.score, reverse=True)
    for rank, s in enumerate(all_scores):
        s.rank = rank + 1

    return SearchResult(
        best_arch=all_scores[0].arch,
        best_score=all_scores[0].score,
        all_scores=all_scores,
        n_evaluated=total,
    )


# ── Export / load ─────────────────────────────────────────────────────────────


def export_best_student(
    supernet: YOLOSupernet,
    result: SearchResult,
    save_path: Path,
) -> None:
    """Save the best sub-architecture as a self-contained checkpoint.

    Checkpoint format::

        {
            "state_dict":      <supernet weights>,
            "arch":            {"backbone_depths": [...], "neck_depths": [...]},
            "teacher_channels": [t_p3, t_p4, t_p5],
            "score":           <best proxy score>,
            "n_evaluated":     <how many architectures were searched>,
            "top5":            [{"rank", "backbone_depths", "neck_depths", "score"}, ...],
        }

    To use at inference time::

        model, arch = load_best_student(save_path)
        model.eval()
        preds, _ = model(x)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    supernet.set_arch(result.best_arch)

    # Infer teacher_channels from the projection conv weight shapes so the
    # checkpoint is self-contained and can be loaded without external config.
    sd = supernet.state_dict()
    teacher_channels = [
        int(sd["proj_p3.weight"].shape[0]),
        int(sd["proj_p4.weight"].shape[0]),
        int(sd["proj_p5.weight"].shape[0]),
    ]

    top5 = [
        {
            "rank": s.rank,
            "backbone_depths": list(s.arch.backbone_depths),
            "neck_depths": list(s.arch.neck_depths),
            "score": s.score,
        }
        for s in result.all_scores[:5]
    ]

    ckpt = {
        "state_dict": sd,
        "arch": {
            "backbone_depths": list(result.best_arch.backbone_depths),
            "neck_depths": list(result.best_arch.neck_depths),
        },
        "teacher_channels": teacher_channels,
        "score": result.best_score,
        "n_evaluated": result.n_evaluated,
        "top5": top5,
    }
    torch.save(ckpt, save_path)
    print(f"[search] best student saved → {save_path}")
    print(
        f"  arch: bd={result.best_arch.backbone_depths} "
        f"nd={result.best_arch.neck_depths}"
    )
    print(
        f"  proxy score: {result.best_score:.4f} "
        f"(evaluated {result.n_evaluated} architectures)"
    )


def load_best_student(ckpt_path: Path) -> tuple[YOLOSupernet, ArchConfig]:
    """Load the best student from an exported checkpoint.

    Returns ``(model, arch_config)`` with weights loaded and arch set.
    Ready for ``model.eval()`` + inference.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    arch = ArchConfig(
        backbone_depths=tuple(ckpt["arch"]["backbone_depths"]),  # type: ignore[arg-type]
        neck_depths=tuple(ckpt["arch"]["neck_depths"]),           # type: ignore[arg-type]
    )
    # Restore teacher_channels from the checkpoint so the model architecture
    # matches the exported weights exactly.
    teacher_channels = tuple(ckpt.get("teacher_channels", [384, 768, 768]))
    model = YOLOSupernet(num_classes=10, teacher_channels=teacher_channels)  # type: ignore[arg-type]
    model.load_state_dict(ckpt["state_dict"])
    model.set_arch(arch)
    return model, arch


# ── Main entry point ──────────────────────────────────────────────────────────


def run_student_search(
    supernet_weights: Optional[str],
    work_dir: Path,
    *,
    n_samples: int = 50,
    search_mode: str = "random",
    imgsz: int = 320,
    batch: int = 4,
    n_proxy_batches: int = 4,
    device: str = "cpu",
    seed: Optional[int] = 42,
    dummy_weights: bool = False,
    wandb_project: str = "distillNas",
    wandb_entity: Optional[str] = None,
    run_name: Optional[str] = None,
    num_classes: int = 10,
) -> SearchResult:
    """Find the best sub-architecture in the trained supernet.

    Parameters
    ----------
    supernet_weights:
        Path to ``supernet_best.pt`` produced by ``distill-supernet``.
        Ignored when ``dummy_weights=True``.
    work_dir:
        Output directory.  Writes ``best_student.pt`` and ``search_results.json``.
    n_samples:
        Architectures to sample (random search only).
    search_mode:
        ``"random"`` (fast, practical) or ``"exhaustive"`` (all 1,296, Kaggle only).
    imgsz:
        Spatial size of proxy validation images.
    batch:
        Batch size for proxy evaluation.
    n_proxy_batches:
        Number of random batches to average over per architecture.
    device:
        Torch device string (``"cpu"`` or ``"0"``/``"cuda:0"``).
    seed:
        Random seed for proxy batches and architecture sampling.
    dummy_weights:
        If True, skip loading supernet weights and use random init.
        Intended for unit tests and smoke tests only.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Build model ───────────────────────────────────────────────────────
    print("[search] initializing supernet...")

    sd: dict | None = None
    if not dummy_weights and supernet_weights:
        weights_path = Path(supernet_weights)
        if weights_path.exists():
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
            # Handle both raw state_dicts and wrapped checkpoints
            if isinstance(ckpt, dict):
                candidate = (
                    ckpt.get("model_state_dict")
                    or ckpt.get("state_dict")
                    or ckpt
                )
            else:
                candidate = ckpt
            if isinstance(candidate, dict) and all(
                isinstance(v, torch.Tensor) for v in candidate.values()
            ):
                sd = candidate
            else:
                print("[search] warning: could not parse checkpoint — using random weights")
        else:
            print(f"[search] warning: {weights_path} not found — using random weights")
    else:
        print("[search] using dummy (random) weights — for testing only")

    # Infer teacher_channels from the state_dict so the model architecture
    # matches the checkpoint's projection convs exactly.
    if sd is not None and "proj_p3.weight" in sd:
        inferred_tc = (
            int(sd["proj_p3.weight"].shape[0]),
            int(sd["proj_p4.weight"].shape[0]),
            int(sd["proj_p5.weight"].shape[0]),
        )
    else:
        inferred_tc = (384, 768, 768)

    model = YOLOSupernet(num_classes=num_classes, teacher_channels=inferred_tc)

    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[search] loaded supernet weights from {weights_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    model.eval()

    # ── Build proxy validation batches ────────────────────────────────────
    print(
        f"[search] building {n_proxy_batches} proxy batches "
        f"(imgsz={imgsz}, batch={batch})..."
    )
    # Resolve device — fall back to cpu if CUDA is unavailable
    if device != "cpu" and not torch.cuda.is_available():
        print("[search] CUDA not available — falling back to CPU")
        device = "cpu"
    dev = torch.device(device if device == "cpu" else f"cuda:{device}")

    # Fixed-seed batches so all architectures are compared on the same inputs
    rng = torch.Generator()
    rng.manual_seed(seed if seed is not None else 0)
    val_batches = [
        torch.randn(batch, 3, imgsz, imgsz, generator=rng)
        for _ in range(n_proxy_batches)
    ]

    # ── Architecture search ───────────────────────────────────────────────
    print(f"[search] running {search_mode} search (seed={seed})...")
    if search_mode == "exhaustive":
        result = exhaustive_search(model, val_batches, dev)
    else:
        result = random_search(model, val_batches, n_samples=n_samples, device=dev, seed=seed)

    # ── Save outputs ──────────────────────────────────────────────────────
    best_student_path = work_dir / "best_student.pt"
    export_best_student(model, result, best_student_path)

    top_n = min(10, len(result.all_scores))
    summary = {
        "best_arch": {
            "backbone_depths": list(result.best_arch.backbone_depths),
            "neck_depths": list(result.best_arch.neck_depths),
        },
        "best_score": result.best_score,
        "n_evaluated": result.n_evaluated,
        "search_mode": search_mode,
        "top10": [
            {
                "rank": s.rank,
                "backbone_depths": list(s.arch.backbone_depths),
                "neck_depths": list(s.arch.neck_depths),
                "score": s.score,
            }
            for s in result.all_scores[:top_n]
        ],
    }
    summary_path = work_dir / "search_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[search] summary saved → {summary_path}")

    # ── Optional W&B logging ──────────────────────────────────────────────
    try:
        import wandb  # type: ignore

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name or "student-search",
            tags=["visdrone", "nas", "search", "spos"],
        )
        wandb.log(
            {
                "search/best_score": result.best_score,
                "search/n_evaluated": result.n_evaluated,
                "search/best_backbone_depths": str(result.best_arch.backbone_depths),
                "search/best_neck_depths": str(result.best_arch.neck_depths),
            }
        )
        artifact = wandb.Artifact("best-student", type="model")
        artifact.add_file(str(best_student_path))
        artifact.add_file(str(summary_path))
        wandb.log_artifact(artifact)
        run.finish()
        print("[search] results logged to W&B")
    except Exception as exc:
        print(f"[search] W&B logging skipped: {exc}")

    return result
