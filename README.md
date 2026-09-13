# Feature Line Rendering for 3D Gaussian Splatting

This repository is an independent reproduction of the **SIGGRAPH Asia 2026
poster _Feature Line Rendering from Rasterization States in 3D Gaussian
Splatting_**. It reconstructs the Lego scene with
[RaDe-GS](https://github.com/HKUST-SAIL/RaDe-GS) and extracts feature lines
directly from information available during Gaussian rasterization, without
first reconstructing a mesh.

The poster does not publish source code or complete equations. The Top-k
distance, thresholds, and field-combination rules in this repository are our
reproduction choices based on the poster diagram. They should not be treated as
the authors' exact implementation.

## Results

All three views are baked into a single looping GIF, side by side, so they are
always frame-synchronized regardless of how the browser loads the page; the
same trained Gaussian model and 360° camera path drive all three panels.

<table>
  <tr>
    <th width="33%">Original 3DGS RGB</th>
    <th width="33%">Feature-line drawing</th>
    <th width="33%">RGB + feature lines</th>
  </tr>
  <tr>
    <td colspan="3"><img src="assets/lego_combined.gif" width="100%" alt="Synchronized rotating RGB, feature-line, and composite renderings"></td>
  </tr>
</table>

The reproduced RaDe-GS model contains 178,171 Gaussians after 30,000 training
iterations. On the 200 held-out NeRF Synthetic Lego views, saved 8-bit RGB
renders achieve 33.01 dB mean PSNR and 0.9753 mean SSIM.

## The central idea: Top-k Gaussian contribution fields

A rendered pixel in 3DGS is not produced by a single primitive. Multiple
projected Gaussians are traversed front to back and alpha-composited. For
Gaussian $g_i$, its contribution to pixel $p$ is

$$
w_i(p) = T_i(p)\,\alpha_i(p),
$$

where $\alpha_i(p)$ is the Gaussian's opacity at the pixel and $T_i(p)$ is
the remaining transmittance before that Gaussian is processed. Ordinary RGB
rendering accumulates these weights with Gaussian colors. This project also
retains the largest $k$ contributions and their global Gaussian IDs:

$$
S_p = \{(g_1,w_1),(g_2,w_2),\ldots,(g_k,w_k)\}.
$$

The ID matters because it tells us whether two neighboring pixels are supported
by the same 3DGS primitives. The weight tells us how much each matched or
unmatched primitive matters. For neighboring pixels $p$ and $q$, retained
weights are normalized independently:

$$
P_g=\frac{w_g(p)}{\sum_{h\in S_p}w_h(p)}, \qquad
Q_g=\frac{w_g(q)}{\sum_{h\in S_q}w_h(q)}.
$$

We align entries by Gaussian ID and measure their shared probability mass:

$$
O(p,q)=\sum_{g\in S_p\cap S_q}\min(P_g,Q_g).
$$

The reproduced Top-k discontinuity is

$$
D_{\mathrm{topk}}(p,q)=1-O(p,q).
$$

For normalized distributions, this is equivalent to total variation distance
over the union of Gaussian IDs. Two pixels can therefore have identical Top-k
ID sets and still be different when their contribution weights change. They can
also differ by only one low-weight ID and remain similar.

Each pixel is compared with its left, right, upper, and lower neighbors. The
largest discontinuity becomes its raw Top-k feature-line value:

$$
E_{\mathrm{topk}}(p)=\max_{q\in N_4(p)}D_{\mathrm{topk}}(p,q).
$$

Thresholding and a smooth transition convert this scalar field into an ink
mask. The implementation is in
[`feature_lines.cu`](external/RaDe-GS/submodules/diff-gaussian-rasterization/feature_lines.cu):

- `topkKernel` replays the rasterizer's sorted splats and records per-pixel IDs
  and weights;
- `fieldsKernel` aligns neighboring Top-k sets and evaluates the discontinuity;
- `renderer.py` applies thresholds, strengths, line dilation, and RGB/ink
  composition.

This remains a screen-space line detector. Its 3DGS-specific input is the
per-pixel Gaussian identity and contribution distribution; it does not construct
persistent curves in 3D. Membership near the k-th position can also change under
small camera movements, so temporal stability remains an open limitation.

## Other rasterization fields

The viewer can combine Top-k discontinuities with four conventional image-space
signals obtained from the same rasterization pass:

- **Opacity:** absolute alpha discontinuity, including the outer silhouette.
- **Depth:** relative depth discontinuity between valid foreground pixels.
- **Normal:** one minus the cosine similarity of neighboring normals.
- **Tone:** absolute luminance difference computed from rasterized RGB.

These fields are useful baselines and complementary cues, but they are not
unique to 3DGS. The Top-k contribution field is the main representation-specific
part of this reproduction. Every field can be enabled independently and its
threshold and strength can be adjusted in the viewer.

## Interactive viewer

Start the local viewer from the repository root:

```powershell
.\scripts\start_viewer.cmd
```

Open <http://127.0.0.1:17865>. The interface provides free orbit, zoom and pan;
RGB, line-only and composite views; individual rasterization fields; pixel-level
Top-k inspection; and separate browser FPS and CUDA timing. RGB, line, and
composite frames are transferred as raw three-channel RGB and uploaded as WebGL
textures. Diagnostic thumbnails use a fixed camera and are generated only once.

The viewer expects the trained model at:

```text
outputs/lego_radegs/point_cloud/iteration_30000/point_cloud.ply
```

## Reproduction

The tested Windows environment uses Python 3.11, PyTorch 2.7.1 with CUDA 12.8,
and Visual Studio 2019, in a Conda environment named `3dgs`. Create a matching
environment, then run:

```powershell
.\scripts\with_3dgs.cmd python scripts\prepare_lego.py
.\scripts\train_lego.cmd
.\scripts\render_lego.cmd
.\scripts\with_3dgs.cmd python scripts\export_lego.py -m outputs/lego_radegs
```

Regenerate the README animations with:

```powershell
.\scripts\with_3dgs.cmd python scripts\export_readme_gifs.py
```

## Repository layout

```text
external/RaDe-GS/          vendored and modified RaDe-GS source
external/fused-ssim/       vendored fused SSIM dependency
viewer/                    Flask, CUDA-backed renderer, and WebGL interface
scripts/                   setup, training, rendering, validation, and GIF tools
assets/                    rotating README demonstrations
```

Datasets, checkpoints, trained PLY files, build products, and generated outputs
are excluded from Git. Follow the reproduction instructions to create them
locally.

## License and attribution

This repository vendors modified RaDe-GS and Gaussian Splatting research code.
The original notices and license files are retained in their corresponding
directories. Gaussian Splatting components are restricted to research and
evaluation use and prohibit commercial use without permission. Review
[`external/RaDe-GS/LICENSE.md`](external/RaDe-GS/LICENSE.md) before
redistribution or use.

The Lego dataset and original Blender asset are not redistributed by this
repository.
