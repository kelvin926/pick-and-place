import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert recorded PNG frames to an MP4 video.")
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--fps", type=float, default=14.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    paths = sorted(frames_dir.glob("frame_*.png"))[:: max(args.stride, 1)]
    if not paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read frame: {paths[0]}")
    if args.width > 0 and first.shape[1] != args.width:
        height = round(first.shape[0] * (args.width / first.shape[1]))
        size = (args.width, height)
    else:
        size = (first.shape[1], first.shape[0])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer: {output}")

    frame_count = 0
    try:
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not read frame: {path}")
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            writer.write(frame)
            frame_count += 1
    finally:
        writer.release()

    print(f"{output.resolve()} {frame_count} frames {args.fps:g} fps")


if __name__ == "__main__":
    main()
