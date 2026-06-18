from visdrone_det.cli import build_parser


def test_benchmark_cli_parses_defaults():
    parser = build_parser()
    args = parser.parse_args(["benchmark-yolov26x"])

    assert args.command == "benchmark-yolov26x"
    assert args.model == "yolo26x.pt"
    assert args.wandb_project == "distillNas"
    assert args.device == "0,1"
