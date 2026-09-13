"""Export RaDe-GS diagnostic fields and summarize held-out RGB renders."""
import json
import sys
from pathlib import Path
from argparse import ArgumentParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/RaDe-GS"))
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.loss_utils import ssim

parser = ArgumentParser()
mp = ModelParams(parser, sentinel=True)
pp = PipelineParams(parser)
args = get_combined_args(parser)
dataset = mp.extract(args)
model_dir = Path(dataset.model_path)
out = model_dir / "previews"
out.mkdir(exist_ok=True)
with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    bg = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.], device="cuda")
    cameras = scene.getTestCameras()
    rows = []
    for idx in (0, 40, 80, 120, 160):
        cam = cameras[idx]
        pkg = render(cam, gaussians, pp.extract(args), bg, dataset.kernel_size)
        fields = {key: pkg[key].cpu().numpy() for key in ("mask", "expected_depth", "median_depth", "normal")}
        assert all(np.isfinite(v).all() for v in fields.values()), "Nonfinite render fields"
        np.savez_compressed(out / f"fields_{idx:05d}.npz", **fields)
        mask = pkg["mask"].squeeze() > .5
        depth = pkg["expected_depth"].squeeze()
        lo, hi = torch.quantile(depth[mask], torch.tensor([.02, .98], device="cuda"))
        depth_vis = ((depth-lo)/(hi-lo).clamp_min(1e-6)).clamp(0, 1)
        depth_vis = torch.where(mask, depth_vis, torch.ones_like(depth_vis))
        normal_vis = torch.where(mask[None], (pkg["normal"]+1)*.5, torch.ones_like(pkg["normal"]))
        panels = [cam.original_image.cuda(), pkg["render"], depth_vis[None].repeat(3, 1, 1), normal_vis]
        row = []
        for name, tensor in zip(("reference", "rgb", "depth", "normal"), panels):
            file = out / f"{name}_{idx:05d}.png"
            torchvision.utils.save_image(tensor, file)
            with Image.open(file) as im:
                row.append(im.convert("RGB").resize((320, 320)))
        rows.append(row)
    sheet = Image.new("RGB", (1280, len(rows)*344), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            sheet.paste(im, (c*320, r*344+24))
            draw.text((c*320+8, r*344+5), ("Reference", "RaDe-GS RGB", "Depth (per-view scaled)", "Normal")[c], fill="black")
    sheet.save(out / "comparison.jpg")
    method = model_dir / "test" / f"ours_{scene.loaded_iter}"
    per_view = {}
    files = sorted((method / "renders").glob("*.png"))
    assert len(files) == len(cameras)
    with imageio.get_writer(out / "test_views.mp4", fps=24, macro_block_size=1) as video:
        for file in files:
            rgb = imageio.imread(file)[..., :3]
            gt = imageio.imread(method / "gt" / file.name)[..., :3]
            x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().cuda()[None]/255
            y = torch.from_numpy(gt.copy()).permute(2, 0, 1).float().cuda()[None]/255
            per_view[file.name] = {"PSNR": (-10*torch.log10((x-y).square().mean())).item(), "SSIM": ssim(x, y).item()}
            video.append_data(rgb)
    summary = {"iteration": scene.loaded_iter, "test_views": len(files), "gaussians": gaussians.get_xyz.shape[0],
               "PSNR": float(np.mean([v["PSNR"] for v in per_view.values()])),
               "SSIM": float(np.mean([v["SSIM"] for v in per_view.values()])),
               "note": "Metrics on saved 8-bit RGB test renders; SSIM uses repository fused implementation. Video follows provided test camera order."}
    (model_dir / "render_metrics.json").write_text(json.dumps(summary, indent=2))
    (model_dir / "per_view_metrics.json").write_text(json.dumps(per_view, indent=2))
    print(json.dumps(summary, indent=2))
