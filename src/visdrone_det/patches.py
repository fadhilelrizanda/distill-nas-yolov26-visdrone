"""Monkey-patches for Ultralytics DDP subprocess W&B integration.

Ultralytics DDP spawns a subprocess via *torch.distributed.run*, re-importing the
trainer class from scratch.  The subprocess does **not** inherit Python objects
from the parent::

    - ``wandb.run`` is ``None``
    - custom callbacks registered via ``model.add_callback()`` are lost
    - ``model.clear_callback()`` affects only the parent's trainer instance

The default Ultralytics ``wb.py`` callbacks fire inside the subprocess and start a
**second unrelated** W&B run, creating noise in the project.

**DDP cannot make W&B API calls** — calling ``wandb.init(resume="must", id=...)``
from inside the DDP subprocess hangs for 90 s then times out with
``CommError`` because you cannot resume a session from a different Unix process on
Kaggle's network setup.

**Solution** — monkey-patch ``ultralytics.utils.dist.generate_ddp_file`` so the
temporary file that the DDP subprocess executes:

1. Strips the default Ultralytics W&B callbacks (prevents duplicate runs)
2. Writes per-epoch / per-checkpoint metrics to a **shared JSONL file** on disk
3. The parent process polls this file in a background thread and forwards to W&B
"""

from __future__ import annotations

import functools
import string
from typing import Any

# ---------------------------------------------------------------------------
# Template injected into the generated DDP temp file.
# Uses string.Template ($var) so Python dict-literals with {} work naturally.
# ---------------------------------------------------------------------------

_INJECTION_TEMPLATE = string.Template(
    """
# === Custom DDP metrics logging (injected by distill-nas/patches.py) ===
import json
import os
from pathlib import Path
from ultralytics.utils import RANK

_METRICS_PATH = os.environ.get("_WANDB_METRICS_FILE", "")

# Strip default Ultralytics W&B callbacks so they do not create a second run.
for _event in ("on_pretrain_routine_start", "on_fit_epoch_end",
               "on_train_epoch_end", "on_train_end"):
    trainer.callbacks[_event] = [
        _cb for _cb in trainer.callbacks.get(_event, [])
        if not getattr(_cb, "__module__", "").endswith(".wb")
    ]

def _write_event(event_dict):
    if not _METRICS_PATH:
        return
    try:
        with open(_METRICS_PATH, "a") as _f:
            _f.write(json.dumps(event_dict) + "\\n")
            _f.flush()
    except Exception:
        pass

# -- per-epoch logging ----------------------------------------------------
def _epoch_log(trainer):
    if RANK not in (-1, 0):
        return
    _write_event({
        "event": "epoch_end",
        "epoch": trainer.epoch + 1,
        "metrics": trainer.metrics,
        "lr": trainer.lr if hasattr(trainer, "lr") and trainer.lr else None,
    })

# -- per-batch live logging (optional, noisy) -----------------------------
if $live_batch_log:

    def _batch_log(trainer):
        if RANK not in (-1, 0):
            return
        step = trainer.epoch * len(trainer.loader) + trainer.batch_i
        _write_event({
            "event": "batch_end",
            "epoch": trainer.epoch + 1,
            "batch": trainer.batch_i,
            "step": step,
            "tloss": trainer.tloss.tolist() if hasattr(trainer.tloss, "tolist") else None,
        })

    trainer.add_callback("on_train_batch_end", _batch_log)

# -- checkpoint logging ---------------------------------------------------
def _checkpoint_log(trainer):
    if RANK not in (-1, 0):
        return
    epoch = trainer.epoch + 1
    if $checkpoint_interval > 0 and epoch % $checkpoint_interval != 0:
        return
    _write_event({
        "event": "checkpoint",
        "epoch": epoch,
        "path": str(trainer.last),
        "metrics": trainer.metrics,
    })

# -- finalise -------------------------------------------------------------
def _train_end(trainer):
    _write_event({"event": "train_end"})

trainer.add_callback("on_fit_epoch_end", _epoch_log)
trainer.add_callback("on_model_save", _checkpoint_log)
trainer.add_callback("on_train_end", _train_end)
"""
)


def patch_generate_ddp_file(
    checkpoint_interval: int = 1,
    live_batch_log: bool = False,
) -> None:
    """Monkey-patch Ultralytics' DDP file generator to inject metrics logging.

    The patched version writes the original generated temp file, then inserts
    *INJECTION_TEMPLATE* into it just before ``trainer.train()`` is called.
    This ensures the custom callbacks are registered *before* training begins.

    Parameters
    ----------
    checkpoint_interval:
        Write a checkpoint event to the metrics file every N epochs (default 1
        = every epoch).  Set to 0 to disable.  The parent process uses these
        events to upload W&B checkpoint artifacts.
    live_batch_log:
        If True, write per-batch loss events to the metrics file (noisy — useful
        for debugging convergence issues).
    """
    import ultralytics.utils.dist as dist_module  # noqa: TID252

    _original = dist_module.generate_ddp_file

    @functools.wraps(_original)
    def _patched(trainer: Any) -> str:
        path = _original(trainer)
        code = _INJECTION_TEMPLATE.substitute(
            checkpoint_interval=checkpoint_interval,
            live_batch_log="True" if live_batch_log else "False",
        )
        # IMPORTANT: insert the injection BEFORE "results = trainer.train()",
        # not at the end of the file — otherwise the callbacks are registered
        # after training complete and never fire.
        with open(path, encoding="utf-8") as f:
            original_content = f.read()
        original_content = original_content.replace(
            "results = trainer.train()",
            f"{code}\nresults = trainer.train()",
            1,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(original_content)
        return path

    dist_module.generate_ddp_file = _patched


def install_patches(
    checkpoint_interval: int = 1,
    live_batch_log: bool = False,
) -> None:
    """Apply all patches needed for DDP-aware metrics logging.

    Call this **after** ``wandb.init()`` and **before** ``model.train()``.
    You must also set ``os.environ["_WANDB_METRICS_FILE"]`` to the path of the
    JSONL file the DDP subprocess will write to (the parent process should
    poll this file and forward events to W&B).

    Parameters
    ----------
    checkpoint_interval:
        Passed through to :func:`patch_generate_ddp_file`.
    live_batch_log:
        Passed through to :func:`patch_generate_ddp_file`.
    """
    patch_generate_ddp_file(
        checkpoint_interval=checkpoint_interval,
        live_batch_log=live_batch_log,
    )
