from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def make_image(path: Path, synthetic: bool, index: int, size: int) -> None:
    image = Image.new("RGB", (size, size), (225, 225, 220) if not synthetic else (35, 35, 35))
    draw = ImageDraw.Draw(image)
    if synthetic:
        for offset in range(0, size, 12):
            draw.line((0, offset, size, size - offset), fill=(220, 95, 70), width=3)
            draw.line((offset, 0, size - offset, size), fill=(70, 170, 145), width=3)
    else:
        draw.ellipse((size // 5, size // 5, size * 4 // 5, size * 4 // 5), fill=(90, 135, 105))
        draw.rectangle((size // 3, size // 2, size * 2 // 3, size * 4 // 5), fill=(70, 95, 150))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def create_demo_data(root: Path, count: int) -> None:
    for index in range(count):
        make_image(root / "real" / f"real_{index:03d}.png", False, index, 224)
        make_image(root / "generator_a" / f"synthetic_{index:03d}.png", True, index, 224)
    print(f"Created {count * 2} demo images below {root.resolve()}")
    print("These images validate the code path only; do not use them for research metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create toy TRACE data for a smoke test")
    parser.add_argument("--output", type=Path, default=Path("data/train"))
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    create_demo_data(args.output, args.count)