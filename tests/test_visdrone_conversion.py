from pathlib import Path

from PIL import Image

from visdrone_det.visdrone import convert_annotation_file


def test_convert_annotation_file_keeps_valid_visdrone_categories(tmp_path: Path):
    image = tmp_path / "000001.jpg"
    Image.new("RGB", (100, 50)).save(image)
    annotation = tmp_path / "000001.txt"
    annotation.write_text(
        "10,5,20,10,1,4,0,0\n"
        "1,1,10,10,1,0,0,0\n"
        "1,1,10,10,1,11,0,0\n",
        encoding="utf-8",
    )

    labels, skipped = convert_annotation_file(annotation, image)

    assert labels == ["3 0.20000000 0.20000000 0.20000000 0.20000000\n"]
    assert skipped == 2
