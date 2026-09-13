"""Render synchronized, looping README demos from the trained Lego model."""
import argparse
import math
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viewer"))
from renderer import LineRenderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--gif-size", type=int, default=384)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()
    if args.frames < 2 or args.gif_size < 64 or args.fps < 1:
        raise ValueError("Invalid animation settings")

    renderer = LineRenderer()
    base = dict(renderer.presets[0])
    base.pop("name", None)
    modes = {"rgb": [], "lines": [], "composite": []}
    options = dict(renderer.info()["defaults"])

    for frame in range(args.frames):
        pose = dict(base)
        pose["yaw"] = base["yaw"] + 2 * math.pi * frame / args.frames
        for mode, images in modes.items():
            rgb8, _, end = renderer.fast_frame({
                "pose": pose,
                "mode": mode,
                "resolution": args.render_size,
                "options": options,
            })
            end.synchronize()
            image = Image.fromarray(rgb8.cpu().numpy(), "RGB")
            if args.gif_size != args.render_size:
                image = image.resize((args.gif_size, args.gif_size), Image.Resampling.LANCZOS)
            images.append(image)
        print(f"Rendered {frame + 1}/{args.frames}", flush=True)

    destination = ROOT / "assets"
    destination.mkdir(exist_ok=True)
    duration = round(1000 / args.fps)
    for mode, images in modes.items():
        path = destination / f"lego_{mode}.gif"
        images[0].save(path, save_all=True, append_images=images[1:], duration=duration,
                       loop=0, optimize=True, disposal=2)
        print(f"Saved {path} ({path.stat().st_size / 1024 / 1024:.2f} MiB)", flush=True)


if __name__ == "__main__":
    main()
