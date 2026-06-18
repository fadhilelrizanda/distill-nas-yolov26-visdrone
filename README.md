# Code Repository

This directory is the structured source repository that should be pushed to GitHub as `distill-nas-yolov26-visdrone`.

Keep reusable experiment logic here:

- Python packages and modules
- Training, evaluation, export, and inference CLIs
- Config files
- Tests
- Lightweight examples

Kaggle notebooks should call this code instead of copying implementation details.

## Expected Shape

```text
code/
├── README.md
├── pyproject.toml
├── src/
├── configs/
├── tests/
└── scripts/
```

## GitHub Contract

This directory is the source of truth for code fetched by Kaggle. If this becomes a nested Git repository, keep generated outputs, datasets, credentials, W&B cache files, and notebook exports out of Git.

Expected remote repository:

```text
distill-nas-yolov26-visdrone
```

## Runtime Contract

Reusable commands should be executable in a fresh Kaggle runtime after installing dependencies and configuring secrets. Prefer a CLI like:

```bash
python -m visdrone_det.train --config configs/example.yaml
```
