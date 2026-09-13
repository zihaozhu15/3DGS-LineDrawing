"""Render a single synchronized, looping README demo GIF from the trained Lego model.

All three views (RGB, lines, composite) are baked into one animated file, side
by side per frame. Separate GIFs would each start their own playback clock the
moment the browser finishes decoding them, so independent load/decode timing
would drift their rotations out of phase; a single file has one playback
clock, so the three panels can never desync.
"""
import argparse
import math
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viewer"))
from renderer import LineRenderer

MODES = ["rgb", "lines", "composite"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--gutter", type=int, default=6)
    args = parser.parse_args()
    if args.frames < 2 or args.size < 64 or args.fps < 1 or args.gutter < 0:
        raise ValueError("Invalid animation settings")

    renderer = LineRenderer()
    base = dict(renderer.presets[0])
    base.pop("name", None)
    options = dict(renderer.info()["defaults"])

    canvas_size = (len(MODES) * args.size + (len(MODES) - 1) * args.gutter, args.size)
    frames = []
    for frame in range(args.frames):
        pose = dict(base)
        pose["yaw"] = base["yaw"] + 2 * math.pi * frame / args.frames
        canvas = Image.new("RGB", canvas_size, "white")
        for i, mode in enumerate(MODES):
            rgb8, _, end = renderer.fast_frame({
                "pose": pose,
                "mode": mode,
                "resolution": args.size,
                "options": options,
            })
            end.synchronize()
            panel = Image.fromarray(rgb8.cpu().numpy(), "RGB")
            canvas.paste(panel, (i * (args.size + args.gutter), 0))
        frames.append(canvas)
        print(f"Rendered {frame + 1}/{args.frames}", flush=True)

    destination = ROOT / "assets"
    destination.mkdir(exist_ok=True)
    duration = round(1000 / args.fps)
    path = destination / "lego_combined.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration,
                   loop=0, optimize=True, disposal=2)
    print(f"Saved {path} ({path.stat().st_size / 1024 / 1024:.2f} MiB)", flush=True)


if __name__ == "__main__":
    main()
