import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert recorded PNG frames to a compact GIF.")
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--duration_ms", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    paths = sorted(frames_dir.glob("frame_*.png"))[:: max(args.stride, 1)]
    if not paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if args.width > 0 and image.width != args.width:
            height = round(image.height * (args.width / image.width))
            image = image.resize((args.width, height), Image.Resampling.LANCZOS)
        images.append(image)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"{output.resolve()} {len(images)} frames")


if __name__ == "__main__":
    main()
