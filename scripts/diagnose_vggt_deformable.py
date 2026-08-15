"""
scripts/diagnose_vggt_deformable.py

Instrumented diagnostic for the 4 items raised before concluding "architecture
underperforms" vs. "there's a bug": fallback-embedding usage rate, VGGT's real input
normalization, image_wh/resolution consistency, and the asymmetric delta_mu range.
Does NOT modify any production file -- reuses the real, live VGGTDeformableController
(pool) and its real sub-modules exactly as built by build_temporal_module(), calling
them individually (same pattern already used to verify gradient checkpointing
earlier this session) to get full intermediate visibility without adding permanent
debug code to deformable_temporal_block.py.

Run: PYTHONNOUSERSITE=1 python scripts/diagnose_vggt_deformable.py
(requires USE_VGGT_DEFORMABLE=True in stage_b_engine.py, same as Gate 1/2)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import torch  # noqa: E402
from mmengine import Config  # noqa: E402

from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402
from src.datasets.occ4dgs_clip_dataset import Occ4DGSClipDataset  # noqa: E402
from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402

from src.training.stage_b_engine import (  # noqa: E402
    REPO_ROOT, GF3D_ROOT, build_pipeline, to_batch_of_one, to_cuda,
    build_stage_a, build_temporal_module, get_real_g0,
    compute_relative_transform, USE_VGGT_DEFORMABLE,
)


def main():
    assert USE_VGGT_DEFORMABLE, "Set USE_VGGT_DEFORMABLE=True in stage_b_engine.py first."

    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        frame_index = json.load(f)
    base_dataset = Occ4DGSDataset(
        nusc, frame_index,
        os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
        os.path.join(REPO_ROOT, "data", "occ3d_gts"),
        pipeline=build_pipeline(),
    )
    clip_dataset = Occ4DGSClipDataset(base_dataset, unroll_window=2)
    scene0061_clip_idx = next(
        i for i, clip in enumerate(clip_dataset.clips)
        if base_dataset.samples[clip[0]][0] == "scene-0061"
    )
    frame0_dict, frame1_dict = clip_dataset[scene0061_clip_idx]
    cuda0 = to_cuda(to_batch_of_one(frame0_dict))
    cuda1 = to_cuda(to_batch_of_one(frame1_dict))

    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_stage_a(cfg)
    pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout, spawn_head = build_temporal_module()
    pool.eval()

    g0, g0_dict = get_real_g0(segmentor, cuda0)
    means_flat = g0.means.squeeze(0) if g0.means.dim() == 3 else g0.means
    rot_flat = g0.rotations.squeeze(0) if g0.rotations.dim() == 3 else g0.rotations
    scale_flat = g0.scales.squeeze(0) if g0.scales.dim() == 3 else g0.scales
    opa_flat = g0.opacities.squeeze(0) if g0.opacities.dim() == 3 else g0.opacities
    sem_flat = g0.semantics.squeeze(0) if g0.semantics.dim() == 3 else g0.semantics
    N = means_flat.shape[0]

    pose_prev = cuda0["metas"]["lidar2global"][0]
    pose_curr = cuda1["metas"]["lidar2global"][0]
    relative_transform = compute_relative_transform(pose_prev, pose_curr)

    print("=" * 70)
    print("ITEM 2: VGGT's actual input normalization (checking the REAL tensor fed in)")
    print("=" * 70)
    vggt_imgs_prev = cuda0["vggt_imgs"]
    vggt_imgs_curr = cuda1["vggt_imgs"]
    print(f"vggt_imgs_prev shape: {tuple(vggt_imgs_prev.shape)}")
    print(f"vggt_imgs_prev value range: [{vggt_imgs_prev.min().item():.6f}, {vggt_imgs_prev.max().item():.6f}]"
          f"  (expect within [0,1] -- raw, NOT ImageNet-normalized)")
    print(f"vggt_imgs_prev mean: {vggt_imgs_prev.mean().item():.4f}, std: {vggt_imgs_prev.std().item():.4f}")
    assert 0.0 <= vggt_imgs_prev.min().item() and vggt_imgs_prev.max().item() <= 1.0 + 1e-4, (
        "vggt_imgs is NOT in [0,1] -- this would mean VGGT is receiving mis-scaled "
        "or already-normalized input, a real bug"
    )
    print("[CONFIRMED] input to VGGT is genuinely raw [0,1] RGB, matching its real "
          "expected contract (vggt/utils/load_fn.py + Aggregator.forward's own "
          "internal normalization)")

    print()
    print("=" * 70)
    print("ITEM 3a: image_wh / resolution consistency (real tensors, not static trace)")
    print("=" * 70)
    image_wh_prev = cuda0["metas"]["vggt_image_wh"]
    image_wh_curr = cuda1["metas"]["vggt_image_wh"]
    print(f"metas['vggt_image_wh'][0,0] = {image_wh_prev[0, 0].tolist()}  (expect [1610.0, 910.0] -- "
          f"NATIVE-PADDED scale, matching projection_mat's own coordinate system, "
          f"NOT the downsampled tensor resolution -- these are deliberately different "
          f"since the image_wh_bug fix)")
    _, N_cam, _, H_actual, W_actual = vggt_imgs_prev.shape
    print(f"vggt_imgs actual (downsampled) spatial shape: ({H_actual}, {W_actual})  (expect (392, 700))")
    assert image_wh_prev[0, 0, 0].item() == 1610.0 and image_wh_prev[0, 0, 1].item() == 910.0, (
        f"metas['vggt_image_wh']={image_wh_prev[0,0].tolist()} is NOT the native-padded "
        f"(1610,910) resolution -- the image_wh_bug fix may not be applied"
    )
    print("[CONFIRMED] metas['vggt_image_wh'] is correctly at NATIVE-PADDED scale "
          "(1610,910) -- matching projection_mat's own coordinate system, distinct "
          "from vggt_imgs' actual downsampled tensor shape, per the image_wh_bug fix")

    with torch.no_grad():
        feat_prev_map, feat_curr_map = pool.vggt(vggt_imgs_prev, vggt_imgs_curr)
    _, _, feat_C, feat_H, feat_W = feat_prev_map.shape
    print(f"feat_prev_map shape: {tuple(feat_prev_map.shape)}  "
          f"(spatial {feat_H}x{feat_W}, expect {H_actual//14}x{W_actual//14})")
    assert feat_H == H_actual // 14 and feat_W == W_actual // 14
    print("[CONFIRMED] VGGT's real output feature map resolution matches "
          "vggt_image_wh/patch_size exactly")

    print()
    print("=" * 70)
    print("ITEM 1 + ITEM 3b: fallback-embedding usage rate, per block, per frame")
    print("=" * 70)
    q = pool.initial_embed(means_flat, rot_flat, scale_flat, opa_flat, sem_flat)
    delta_mu = torch.zeros(N, 3, device=means_flat.device, dtype=means_flat.dtype)
    delta_r = torch.zeros(N, 4, device=means_flat.device, dtype=means_flat.dtype)
    delta_r[:, 0] = 1.0

    projection_mat_prev = cuda0["metas"]["projection_mat"]
    projection_mat_curr = cuda1["metas"]["projection_mat"]

    from model.encoder.gaussian_encoder.deformable_module_3d import DeformableFeatureAggregation3D

    for block_idx, block in enumerate(pool.blocks):
        with torch.no_grad():
            key_points_prev = means_flat.unsqueeze(0).unsqueeze(2)
            _, mask_prev = DeformableFeatureAggregation3D.project_points_3d(
                key_points_prev, projection_mat_prev, block.d_bound, image_wh_prev
            )
            mask_prev = mask_prev[:, 0, :, 0]  # (num_cam, N)

            means_curr_estimate = means_flat + delta_mu
            ones = torch.ones(N, 1, device=means_flat.device, dtype=means_flat.dtype)
            means_h = torch.cat([means_curr_estimate, ones], dim=-1)
            means_curr_frame = (relative_transform.to(means_flat.dtype) @ means_h.T).T[:, :3]
            key_points_curr = means_curr_frame.unsqueeze(0).unsqueeze(2)
            _, mask_curr = DeformableFeatureAggregation3D.project_points_3d(
                key_points_curr, projection_mat_curr, block.d_bound, image_wh_curr
            )
            mask_curr = mask_curr[:, 0, :, 0]

            valid_prev = mask_prev.sum(dim=0) > 0  # (N,) -- True if >=1 camera sees it
            valid_curr = mask_curr.sum(dim=0) > 0
            fallback_prev_pct = 100.0 * (~valid_prev).float().mean().item()
            fallback_curr_pct = 100.0 * (~valid_curr).float().mean().item()
            avg_cams_prev = mask_prev.sum(dim=0).float().mean().item()
            avg_cams_curr = mask_curr.sum(dim=0).float().mean().item()

            print(f"Block {block_idx}: prev-frame fallback rate = {fallback_prev_pct:.2f}% "
                  f"(avg {avg_cams_prev:.2f} cams/Gaussian), "
                  f"curr-frame fallback rate = {fallback_curr_pct:.2f}% "
                  f"(avg {avg_cams_curr:.2f} cams/Gaussian)")

            # Step the block forward for real (needed to get delta_mu/delta_r/q for the next block)
            delta_mu, delta_r, q = block(
                means_flat, delta_mu, delta_r, q, relative_transform,
                projection_mat_prev, image_wh_prev, feat_prev_map,
                projection_mat_curr, image_wh_curr, feat_curr_map,
            )

    print()
    print("=" * 70)
    print("ITEM 3c: delta_mu distribution (mean, not just min/max) after block 4")
    print("=" * 70)
    print(f"delta_mu: mean={delta_mu.mean(dim=0).tolist()}, "
          f"std={delta_mu.std(dim=0).tolist()}")
    print(f"delta_mu per-axis min/max: "
          f"x=[{delta_mu[:,0].min().item():.3f},{delta_mu[:,0].max().item():.3f}] "
          f"y=[{delta_mu[:,1].min().item():.3f},{delta_mu[:,1].max().item():.3f}] "
          f"z=[{delta_mu[:,2].min().item():.3f},{delta_mu[:,2].max().item():.3f}]")
    print("(a genuinely centered-near-zero MEAN with a wide min/max range is expected/"
          "healthy at init -- DeformHeadMu's own final layer is NOT zero-initialized, "
          "unlike offset_mlp/attn_mlp, so per-Gaussian outputs vary; an asymmetric mean "
          "clearly offset from 0 would be the real signal to worry about, not the range)")

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
