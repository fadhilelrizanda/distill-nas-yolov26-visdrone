"""SPOS supernet student model for VisDrone NAS + knowledge distillation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn

# ── Search space ───────────────────────────────────────────────────────────

BACKBONE_DEPTH_CHOICES: list[int] = [1, 2, 3]
NECK_DEPTH_CHOICES: list[int] = [1, 2]

# Fixed channel widths (≈ YOLOv26s scale)
_STEM_CH = 32
_S1_CH = 64
_S2_CH = 128
_S3_CH = 256
_S4_CH = 512

# Student neck output channels per scale
_N_P3 = 128
_N_P4 = 256
_N_P5 = 512

# Default teacher FPN channel counts — overridden by _find_fpn_layers() in distill.py
_DEFAULT_T_P3 = 384
_DEFAULT_T_P4 = 768
_DEFAULT_T_P5 = 768


@dataclass
class ArchConfig:
    """A concrete sub-architecture sampled from the SPOS supernet."""

    backbone_depths: tuple[int, int, int, int]  # stages 1-4; each ∈ BACKBONE_DEPTH_CHOICES
    neck_depths: tuple[int, int, int, int]  # [fpn_p4, fpn_p3, pan_p4, pan_p5]; each ∈ NECK_DEPTH_CHOICES


# ── Building blocks ────────────────────────────────────────────────────────


class _Conv(nn.Module):
    """Conv2d + BatchNorm2d + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _Bottleneck(nn.Module):
    """Standard CSP bottleneck (3×3 → 3×3) with optional residual add."""

    def __init__(self, ch: int, shortcut: bool = True) -> None:
        super().__init__()
        self.cv1 = _Conv(ch, ch, 3)
        self.cv2 = _Conv(ch, ch, 3)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.shortcut else out


class SearchableC2f(nn.Module):
    """C2f with runtime-configurable bottleneck depth for SPOS weight sharing.

    ``cv2`` is always sized to accept ``max_n`` bottleneck outputs.  Inactive
    depth slots are zero-filled so gradients flow only through the active path,
    while all shared weights participate across the depth choices that activate
    them.

    Invariant: ``1 <= _active_n <= max_n``.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        max_n: int,
        shortcut: bool = True,
    ) -> None:
        super().__init__()
        hidden = out_ch // 2
        self._hidden = hidden
        self._max_n = max_n
        self._active_n: int = max_n
        self.cv1 = _Conv(in_ch, 2 * hidden, 1)
        self.m = nn.ModuleList(_Bottleneck(hidden, shortcut) for _ in range(max_n))
        # cv2 always accepts (2 + max_n) * hidden channels regardless of active depth
        self.cv2 = _Conv((2 + max_n) * hidden, out_ch, 1)

    def set_active_n(self, n: int) -> None:
        if n < 1 or n > self._max_n:
            raise ValueError(f"active depth {n} out of valid range [1, {self._max_n}]")
        self._active_n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))  # [x0, x1], each [B, hidden, H, W]
        for i in range(self._max_n):
            if i < self._active_n:
                y.append(self.m[i](y[-1]))
            else:
                # Zero-fill: no computation, no gradient through inactive slots
                y.append(torch.zeros_like(y[1]))
        return self.cv2(torch.cat(y, 1))


class _SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast: three cascaded MaxPool2d(k=5)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        hidden = in_ch // 2
        self.cv1 = _Conv(in_ch, hidden, 1)
        self.cv2 = _Conv(hidden * 4, out_ch, 1)
        self.pool = nn.MaxPool2d(5, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], 1))


class _DecoupledHead(nn.Module):
    """Anchor-free decoupled detection head for one FPN scale."""

    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__()
        self.cls_branch = nn.Sequential(
            _Conv(in_ch, in_ch, 3),
            nn.Conv2d(in_ch, num_classes, 1),
        )
        self.box_branch = nn.Sequential(
            _Conv(in_ch, in_ch, 3),
            nn.Conv2d(in_ch, 4, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cls_branch(x), self.box_branch(x)


# ── Supernet ───────────────────────────────────────────────────────────────


class YOLOSupernet(nn.Module):
    """Single-Path One-Shot supernet for YOLO-style detection on VisDrone.

    Architecture mirrors YOLOv26s channel widths with searchable C2f depths:

      Backbone: stem → 4× SearchableC2f stages → SPPF
      Neck:     FPN top-down (P5→P4→P3) + PAN bottom-up (P3→P4→P5) via SearchableC2f
      Head:     decoupled anchor-free at P3/P4/P5 (strides 8/16/32)

    Search space:
      backbone stages 1-4: depth ∈ BACKBONE_DEPTH_CHOICES = [1, 2, 3]
      neck stages [fpn_p4, fpn_p3, pan_p4, pan_p5]: depth ∈ NECK_DEPTH_CHOICES = [1, 2]
      total sub-networks: 3**4 × 2**4 = 1,296

    Projection convs ``proj_p3/p4/p5`` map student FPN features to teacher
    channel dimensions for MSE-based feature knowledge distillation.
    """

    def __init__(
        self,
        num_classes: int = 10,
        teacher_channels: tuple[int, int, int] = (
            _DEFAULT_T_P3,
            _DEFAULT_T_P4,
            _DEFAULT_T_P5,
        ),
    ) -> None:
        super().__init__()
        nc = num_classes
        t_p3, t_p4, t_p5 = teacher_channels

        # ── Backbone ──────────────────────────────────────────────────────
        self.stem = _Conv(3, _STEM_CH, 3, 2)                          # P1/2:  320×320
        self.ds1 = _Conv(_STEM_CH, _S1_CH, 3, 2)                     # P2/4:  160×160
        self.bb0 = SearchableC2f(_S1_CH, _S1_CH, max_n=3)            # stage 1
        self.ds2 = _Conv(_S1_CH, _S2_CH, 3, 2)                       # P3/8:   80×80
        self.bb1 = SearchableC2f(_S2_CH, _S2_CH, max_n=3)            # stage 2 → bb_P3
        self.ds3 = _Conv(_S2_CH, _S3_CH, 3, 2)                       # P4/16:  40×40
        self.bb2 = SearchableC2f(_S3_CH, _S3_CH, max_n=3)            # stage 3 → bb_P4
        self.ds4 = _Conv(_S3_CH, _S4_CH, 3, 2)                       # P5/32:  20×20
        self.bb3 = SearchableC2f(_S4_CH, _S4_CH, max_n=3)            # stage 4
        self.sppf = _SPPF(_S4_CH, _S4_CH)

        # ── Neck FPN top-down ─────────────────────────────────────────────
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck0 = SearchableC2f(_S4_CH + _S3_CH, _N_P4, max_n=2)  # td_P4: 40×40

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck1 = SearchableC2f(_N_P4 + _S2_CH, _N_P3, max_n=2)   # td_P3: 80×80 → student P3

        # ── Neck PAN bottom-up ────────────────────────────────────────────
        self.dconv1 = _Conv(_N_P3, _N_P3, 3, 2)
        self.neck2 = SearchableC2f(_N_P3 + _N_P4, _N_P4, max_n=2)    # student P4: 40×40

        self.dconv2 = _Conv(_N_P4, _N_P4, 3, 2)
        self.neck3 = SearchableC2f(_N_P4 + _S4_CH, _N_P5, max_n=2)   # student P5: 20×20

        # ── Detection heads ───────────────────────────────────────────────
        self.head_p3 = _DecoupledHead(_N_P3, nc)  # stride 8
        self.head_p4 = _DecoupledHead(_N_P4, nc)  # stride 16
        self.head_p5 = _DecoupledHead(_N_P5, nc)  # stride 32

        # ── Projection convs for feature distillation ─────────────────────
        self.proj_p3 = nn.Conv2d(_N_P3, t_p3, 1, bias=False)
        self.proj_p4 = nn.Conv2d(_N_P4, t_p4, 1, bias=False)
        self.proj_p5 = nn.Conv2d(_N_P5, t_p5, 1, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        prior_prob = 0.01
        cls_bias_init = math.log((1.0 - prior_prob) / prior_prob)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Suppress false positives early by biasing cls heads low
        for head in (self.head_p3, self.head_p4, self.head_p5):
            cls_linear = head.cls_branch[-1]
            if isinstance(cls_linear, nn.Conv2d) and cls_linear.bias is not None:
                nn.init.constant_(cls_linear.bias, cls_bias_init)

    # ── Architecture sampling ─────────────────────────────────────────────

    def sample_arch(self) -> ArchConfig:
        """Return a uniformly random sub-architecture from the search space."""
        bd = tuple(random.choice(BACKBONE_DEPTH_CHOICES) for _ in range(4))
        nd = tuple(random.choice(NECK_DEPTH_CHOICES) for _ in range(4))
        return ArchConfig(backbone_depths=bd, neck_depths=nd)  # type: ignore[arg-type]

    def set_arch(self, arch: ArchConfig) -> None:
        """Activate the sub-architecture described by *arch*.

        Backbone ordering: ``arch.backbone_depths[0..3]`` → stages 1-4 (bb0..bb3).
        Neck ordering: ``arch.neck_depths[0..3]`` → [neck0, neck1, neck2, neck3]
        which correspond to [fpn_p4, fpn_p3, pan_p4, pan_p5].
        """
        for block, n in zip(
            (self.bb0, self.bb1, self.bb2, self.bb3),
            arch.backbone_depths,
        ):
            block.set_active_n(n)
        for block, n in zip(
            (self.neck0, self.neck1, self.neck2, self.neck3),
            arch.neck_depths,
        ):
            block.set_active_n(n)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Run the currently active sub-architecture.

        Parameters
        ----------
        x : torch.Tensor
            Input images ``[B, 3, H, W]``.

        Returns
        -------
        preds : list of 3 ``(cls_logits, box_deltas)`` tuples
            ``cls_logits [B, nc, Hs, Ws]``, ``box_deltas [B, 4, Hs, Ws]``
            for scales P3/P4/P5 (strides 8/16/32).
        student_feats : tuple of 3 tensors
            ``(proj_p3, proj_p4, proj_p5)`` — student FPN features projected
            to teacher channel dimensions for MSE-based distillation.
        """
        # Backbone
        x = self.stem(x)
        x = self.ds1(x)
        x = self.bb0(x)
        x = self.ds2(x)
        bb_p3 = self.bb1(x)          # [B, 128,  80, 80]
        x = self.ds3(bb_p3)
        bb_p4 = self.bb2(x)          # [B, 256,  40, 40]
        x = self.ds4(bb_p4)
        x = self.bb3(x)
        sppf_out = self.sppf(x)      # [B, 512,  20, 20]

        # Neck FPN top-down
        td_p4 = self.neck0(torch.cat([self.up1(sppf_out), bb_p4], 1))  # [B, 256, 40, 40]
        p3_feat = self.neck1(torch.cat([self.up2(td_p4), bb_p3], 1))   # [B, 128, 80, 80]

        # Neck PAN bottom-up
        p4_feat = self.neck2(torch.cat([self.dconv1(p3_feat), td_p4], 1))    # [B, 256, 40, 40]
        p5_feat = self.neck3(torch.cat([self.dconv2(p4_feat), sppf_out], 1)) # [B, 512, 20, 20]

        preds = [
            self.head_p3(p3_feat),
            self.head_p4(p4_feat),
            self.head_p5(p5_feat),
        ]

        student_feats = (
            self.proj_p3(p3_feat),
            self.proj_p4(p4_feat),
            self.proj_p5(p5_feat),
        )

        return preds, student_feats
