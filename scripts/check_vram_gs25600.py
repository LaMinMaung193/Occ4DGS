"""
scripts/check_vram_gs25600.py

Step 1 of the full-dataset plan (EXPERIMENT_LOG.md): cheapest possible check of
whether N_g=25600 (GaussianFormer3D's real full-scale config) fits in our RTX 3090's
24GB VRAM, BEFORE committing to wiring up the real SurroundOcc data pipeline or a
real training run. Uses synthetic tensors matching the real config's actual shapes
(1600x900 images, N_g=25600) -- not real data, since that's Step 2's job.

Two forward passes measured separately:
  1. rep_only=True -- Stage A's G_0 generation alone (encoder only, no splat/head).
  2. Full forward -- encoder + GaussianHead splat, matching a real eval/inference step.

HONEST CAVEAT: this measures FORWARD PASS ONLY, no backward/optimizer. A real
training step will use MORE VRAM than this reports (gradients + optimizer states +
activations kept for backprop). Treat a "fits comfortably" result here as a
necessary-but-not-sufficient signal, not a guarantee training itself will fit.

Run from repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/check_vram_gs25600.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from mmengine import Config  # noqa: E402

from src.datasets.gf3d_pipeline import GF3D_ROOT  # noqa: E402 -- triggers sys.path/chdir side effect

from mmseg.models import build_segmentor  # noqa: E402
import model  # noqa: E402,F401 -- triggers @MODELS.register_module() decorators


REAL_CONFIG = os.path.join(GF3D_ROOT, "config", "nuscenes_surroundocc_gs25600.py")
IMG_H, IMG_W = 928, 1600  # 928, not the raw 900 -- GF3D's real pipeline pads height
                          # to the next multiple of 32 (depth head's downsample
                          # factor) before it reaches the model; confirmed directly
                          # from depth_head.py's own comment ("H/ds=928/32=29").
                          # Width (1600) is already exactly divisible by 32, no
                          # padding needed there. The config's input_shape=(1600,900)
                          # is the RAW pre-padding size, not what the model receives.
N_CAMS = 6
N_G = 25600


def make_synthetic_batch(cfg):
    imgs = torch.randn(1, N_CAMS, 3, IMG_H, IMG_W)
    projection_mat = torch.eye(4).view(1, 1, 4, 4).repeat(1, N_CAMS, 1, 1)
    image_wh = torch.tensor([IMG_W, IMG_H], dtype=torch.float32).view(1, 1, 2).repeat(1, N_CAMS, 1)

    # GaussianHead's splat/occupancy-eval step (confirmed via real KeyError, not
    # assumed) needs occ_xyz/occ_label/occ_cam_mask -- same convention already
    # established in our own project's own pipeline (to_batch_of_one(),
    # LoadOccupancyOcc3d). Grid is H=200,W=200,D=16 (confirmed from this config's
    # own head.cuda_kwargs). IMPORTANT: shape must be the UN-flattened grid
    # (B,H,W,D,...) -- GaussianHead._sampling() does its own internal
    # gt_xyz.flatten(1,3), confirmed from the real IndexError this produced when
    # we pre-flattened it ourselves. Matches our own project's own
    # LoadOccupancyOcc3d.get_meshgrid() output shape exactly, not a new guess.
    H_grid, W_grid, D_grid = 200, 200, 16
    pc_range = cfg.model["encoder"]["deformable_model"]["kps_generator"]["pc_range"]
    occ_xyz = torch.rand(1, H_grid, W_grid, D_grid, 3)
    for i in range(3):
        occ_xyz[..., i] = occ_xyz[..., i] * (pc_range[i + 3] - pc_range[i]) + pc_range[i]
    occ_label = torch.randint(0, 18, (1, H_grid, W_grid, D_grid))  # 18 = semantic_dim(17) + empty(1)
    occ_cam_mask = torch.ones(1, H_grid, W_grid, D_grid, dtype=torch.bool)  # synthetic: all visible

    metas = {
        "projection_mat": projection_mat,
        "image_wh": image_wh,
        "occ_xyz": occ_xyz,
        "occ_label": occ_label,
        "occ_cam_mask": occ_cam_mask,
    }
    # Synthetic LiDAR points -- shape doesn't need to be realistic for a VRAM check,
    # just present and on the right device (voxelize_lidar consumes this).
    points = [torch.randn(20000, 5)]
    # DepthHead_GTDpt (confirmed from real source, depth_head.py forward()) requires
    # real ground-truth depth, shape (B, N, 1, H, W) before its own squeeze(2) --
    # synthetic here since real depth GT isn't yet available (Step 2, unresolved --
    # readme.md documents a real download link, not yet obtained). Values are
    # meaningless (VRAM-only check), but the SHAPE must be correct or forward()
    # crashes on incompatible tensor ops, not just wrong numbers.
    dpt = torch.randn(1, N_CAMS, 1, IMG_H, IMG_W)
    return imgs, metas, points, dpt


def report_vram(label):
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  [{label}] peak VRAM so far: {peak_gb:.2f} GB")
    return peak_gb


def main():
    print(f"Loading real config: {REAL_CONFIG}")
    cfg = Config.fromfile(REAL_CONFIG)
    print(f"Confirmed N_g (num_anchor) = {cfg.model['lifter']['num_anchor']}")
    assert cfg.model["lifter"]["num_anchor"] == N_G

    print("\nBuilding model (random init -- NOT loading any checkpoint, this is a "
          "pure architecture/VRAM check)...")
    segmentor = build_segmentor(cfg.model)
    segmentor.init_weights()
    segmentor = segmentor.cuda()
    segmentor.eval()

    print("\n=== Forward pass 1: rep_only=True (Stage A G_0 generation only) ===")
    torch.cuda.reset_peak_memory_stats()
    imgs, metas, points, dpt = make_synthetic_batch(cfg)
    imgs, dpt = imgs.cuda(), dpt.cuda()
    points_cuda = [p.cuda() for p in points]
    metas_cuda = {k: v.cuda() for k, v in metas.items()}
    with torch.no_grad():
        representation = segmentor(imgs=imgs, metas=metas_cuda, points=points_cuda, dpt=dpt, rep_only=True)
    g0 = representation[-1]["gaussian"]
    print(f"  G_0 means shape: {tuple(g0.means.shape)} (expect (1, {N_G}, 3))")
    assert g0.means.shape[1] == N_G
    peak_rep_only = report_vram("rep_only")

    print("\n=== Forward pass 2: full forward (encoder + GaussianHead splat) ===")
    # Fresh tensors here, deliberately -- extract_img_dpt_feat's real code does
    # imgs.squeeze_(0)/dpt.squeeze_(0), IN-PLACE (confirmed: trailing underscore
    # really does mutate permanently). Reusing pass 1's already-squeezed tensors
    # for pass 2 would silently corrupt shapes -- this is a test-harness bug we
    # found and fixed, not a GF3D bug.
    torch.cuda.reset_peak_memory_stats()
    imgs, metas, points, dpt = make_synthetic_batch(cfg)
    imgs, dpt = imgs.cuda(), dpt.cuda()
    points_cuda = [p.cuda() for p in points]
    metas_cuda = {k: v.cuda() for k, v in metas.items()}
    with torch.no_grad():
        out = segmentor(imgs=imgs, metas=metas_cuda, points=points_cuda, dpt=dpt)
    print(f"  Output keys: {list(out.keys())}")
    peak_full = report_vram("full forward")

    print(f"\n=== SUMMARY ===")
    print(f"N_g=25600, image size {IMG_W}x{IMG_H}, batch_size=1:")
    print(f"  Peak VRAM, rep_only (G_0 generation):      {peak_rep_only:.2f} GB")
    print(f"  Peak VRAM, full forward (+ splat/head):    {peak_full:.2f} GB")
    print(f"  Available on RTX 3090:                     24.00 GB")
    print(f"\nCAVEAT: forward-pass-only. Real training (backward + optimizer + "
          f"grad-accum) will use MORE than the 'full forward' number above -- "
          f"this result answers 'does the model itself fit,' not 'will training fit.'")


if __name__ == "__main__":
    main()
