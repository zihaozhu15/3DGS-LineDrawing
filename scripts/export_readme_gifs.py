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
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--render-size", type=int, default=768)
    parser.add_argument("--gif-size", type=int, default=600)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--gutter", type=int, default=4)
    parser.add_argument("--colors", type=int, default=256)
    args = parser.parse_args()
    if args.frames < 2 or args.gif_size < 64 or args.fps < 1 or args.gutter < 0 or not (2 <= args.colors <= 256):
        raise ValueError("Invalid animation settings")

    renderer = LineRenderer()
    base = dict(renderer.presets[0])
    base.pop("name", None)
    options = dict(renderer.info()["defaults"])

    canvas_size = (len(MODES) * args.gif_size + (len(MODES) - 1) * args.gutter, args.gif_size)
    frames = []
    for frame in range(args.frames):
        pose = dict(base)
        pose["yaw"] = base["yaw"] + 2 * math.pi * frame / args.frames
        canvas = Image.new("RGB", canvas_size, "white")
        for i, mode in enumerate(MODES):
            rgb8, _, end = renderer.fast_frame({
                "pose": pose,
                "mode": mode,
                "resolution": args.render_size,
                "options": options,
            })
            end.synchronize()
            panel = Image.fromarray(rgb8.cpu().numpy(), "RGB")
            if args.gif_size != args.render_size:
                panel = panel.resize((args.gif_size, args.gif_size), Image.Resampling.LANCZOS)
            canvas.paste(panel, (i * (args.gif_size + args.gutter), 0))
        frames.append(canvas)
        print(f"Rendered {frame + 1}/{args.frames}", flush=True)

    destination = ROOT / "assets"
    destination.mkdir(exist_ok=True)
    duration = round(1000 / args.fps)
    path = destination / "lego_combined.gif"
    if args.colors < 256:
        shared_palette = frames[0].quantize(colors=args.colors, method=Image.Quantize.FASTOCTREE)
        frames = [f.quantize(palette=shared_palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration,
                   loop=0, optimize=True, disposal=2)
    print(f"Saved {path} ({path.stat().st_size / 1024 / 1024:.2f} MiB)", flush=True)


if __name__ == "__main__":
    main()
