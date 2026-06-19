"""Monkey-patches for Ultralytics DDP subprocess W&B integration.

Ultralytics DDP spawns a subprocess via *torch.distributed.run*, re-importing the
trainer class from scratch.  The subprocess does **not** inherit Python objects
from the parent::

    - ``wandb.run`` is ``None``
    - custom callbacks registered via ``model.add_callback()`` are lost
    - ``model.clear_callback()`` affects only the parent's trainer instance

The default Ultralytics ``wb.py`` callbacks fire inside the subprocess and start
a **second unrelated** W&B run, creating noise in the project.

**Solution** — monkey-patch ``ultralytics.utils.dist.generate_ddp_file`` so the
temporary file that the DDP subprocess executes includes custom W&B callbacks
that resume the parent process's run.
"""

from __future__ import annotations

import functools
import string
from typing import Any

# ---------------------------------------------------------------------------
# Template appended to the generated DDP temp file.
# Uses string.Template ($var) so Python dict-literals with {} work naturally.
# ---------------------------------------------------------------------------

_INJECTION_TEMPLATE = string.Template(
    """
# === Custom W&B DDP callbacks (injected by distill-nas/patches.py) ===
import os
from pathlib import Path

_WANDB_DDP_RUN_ID = os.environ.get("_WANDB_DDP_RUN_ID")
_WANDB_DDP_PROJECT = os.environ.get("_WANDB_DDP_PROJECT", "")
_WANDB_DDP_ENTITY = os.environ.get("_WANDB_DDP_ENTITY")
_CHECKPOINT_INTERVAL = $checkpoint_interval
_LIVE_BATCH_LOG = $live_batch_log

if _WANDB_DDP_RUN_ID:
    import wandb as _wb

    # Strip default Ultralytics W&B callbacks from the subprocess trainer so
    # they do not create a second unrelated run.  We filter by __module__
    # because the callbacks are plain functions defined in wb.py.
    for _event in ("on_pretrain_routine_start", "on_fit_epoch_end",
                   "on_train_epoch_end", "on_train_end"):
        trainer.callbacks[_event] = [
            _cb for _cb in trainer.callbacks.get(_event, [])
            if not getattr(_cb, "__module__", "").endswith(".wb")
        ]

    def _ensure_run():
        if _wb.run is None:
            _wb.init(project=_WANDB_DDP_PROJECT,
                     entity=_WANDB_DDP_ENTITY,
                     id=_WANDB_DDP_RUN_ID,
                     resume="must")

    # -- per-epoch logging ------------------------------------------------
    def _epoch_log(trainer):
        if trainer.rank not in (-1, 0):
            return
        _ensure_run()
        epoch = trainer.epoch + 1
        log = {"epoch": epoch, **trainer.metrics}
        if hasattr(trainer, "lr") and trainer.lr:
            log.update(trainer.lr)
        _wb.log(log, step=epoch)

    # -- per-batch live logging (optional, noisy) -------------------------
    if _LIVE_BATCH_LOG:

        def _batch_log(trainer):
            if trainer.rank not in (-1, 0):
                return
            _ensure_run()
            step = trainer.epoch * len(trainer.train_loader) + trainer.batch_i
            loss_items = getattr(trainer, "tloss", None)
            log = {"live/step": step, "live/epoch": trainer.epoch + 1}
            if loss_items is not None:
                if hasattr(loss_items, "tolist"):
                    items = loss_items.tolist()
                    if isinstance(items, (list, tuple)):
                        for idx, val in enumerate(items):
                            log["live/loss_" + str(idx)] = val
                    else:
                        log["live/loss"] = items
            _wb.log(log, step=step)

        trainer.add_callback("on_train_batch_end", _batch_log)

    # -- checkpoint artifact logging --------------------------------------
    def _checkpoint_log(trainer):
        if trainer.rank not in (-1, 0):
            return
        epoch = trainer.epoch + 1
        if _CHECKPOINT_INTERVAL > 0 and epoch % _CHECKPOINT_INTERVAL != 0:
            return
        _ensure_run()
        last_pt = Path(trainer.last)
        if last_pt.exists():
            art = _wb.Artifact(
                name="checkpoint-epoch%04d" % epoch,
                type="model",
                metadata={"epoch": epoch, **trainer.metrics},
            )
            art.add_file(str(last_pt), name="last.pt")
            _wb.log_artifact(art)

    # -- finalise ---------------------------------------------------------
    def _train_end(trainer):
        _ensure_run()
        _wb.run.finish()

    # Register callbacks (order matters: epoch_end runs AFTER model_save in
    # the Ultralytics loop, so metrics are up-to-date).
    trainer.add_callback("on_fit_epoch_end", _epoch_log)
    trainer.add_callback("on_model_save", _checkpoint_log)
    trainer.add_callback("on_train_end", _train_end)
else:
    import warnings
    warnings.warn("_WANDB_DDP_RUN_ID not set -- no W&B DDP logging in subprocess")
"""
)


def patch_generate_ddp_file(
    checkpoint_interval: int = 1,
    live_batch_log: bool = False,
) -> None:
    """Monkey-patch Ultralytics' DDP file generator to inject W&B callbacks.

    The patched version writes the original generated temp file, then inserts
    *INJECTION_TEMPLATE* into it just before ``trainer.train()`` is called.
    This ensures the custom W&B callbacks are registered *before* training begins.

    Parameters
    ----------
    checkpoint_interval:
        Upload a W&B checkpoint artifact every N epochs (default 1 = every
        epoch).  Set to 0 to disable checkpoint artifacts during training
        (the final checkpoint is still uploaded by the parent process).
    live_batch_log:
        If True, log per-batch training losses (noisy — useful for
        debugging convergence issues).
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
    """Apply all patches needed for DDP-aware W&B logging.

    Call this **after** ``wandb.init()`` and **before** ``model.train()``.
    Also set ``os.environ["_WANDB_DDP_RUN_ID"]`` to the current run id so the
    DDP subprocess knows which run to resume.

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
