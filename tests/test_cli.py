from visdrone_det.cli import main


def test_train_placeholder_runs(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("project: distillNas
", encoding="utf-8")

    assert main(["train", "--config", str(config), "--output-dir", str(tmp_path / "out")]) == 0
