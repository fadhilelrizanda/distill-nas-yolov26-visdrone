"""Command-line entrypoints for VisDrone experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visdrone-det")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Run a training experiment")
    train.add_argument("--config", type=Path, required=True, help="Path to experiment config")
    train.add_argument("--data-root", type=Path, default=Path("/kaggle/input/visdrone-dataset"))
    train.add_argument("--wandb-project", default="distillNas")
    train.add_argument("--output-dir", type=Path, default=Path("outputs"))

    return parser


def run_train(args: argparse.Namespace) -> int:
    # Placeholder until the training stack is implemented.
    print(f"config={args.config}")
    print(f"data_root={args.data_root}")
    print(f"wandb_project={args.wandb_project}")
    print(f"output_dir={args.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return run_train(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
