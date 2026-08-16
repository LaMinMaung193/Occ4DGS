"""
scripts/diagnose_vggt_quat_and_offsets.py

Two remaining pieces not yet directly verified on real data/weights, per Liam's
request to confirm every step of this design is working as intended before drawing
any conclusion: (1) is delta_r behaving like a sane, correctly-bounded rotation after
real training (valid unit quaternions, angle within max_angle_rad), and (2) are the
K=4 deformable offsets -- now that the symmetry-breaking fix lets them actually
learn -- moving to sensible values, or exploding/collapsing to something degenerate?

Run: PYTHONNOUSERSITE=1 python scripts/diagnose_vggt_quat_and_offsets.py
(requires USE_VGGT_DEFORMABLE=True, and a saved checkpoint from a completed run)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import math  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
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

    ckpt_path = os.path.join(
        REPO_ROOT, "experiments", "stage_b_temporal_checkpoints", "stage1_warmup_temporal_n3_final.pth"
    )
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cuda")
    pool.load_state_dict(ckpt["pool"])
    pool.eval()
    print(f"Loaded trained checkpoint from {ckpt_path}\n")

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

    projection_mat_prev = cuda0["metas"]["projection_mat"]
    projection_mat_curr = cuda1["metas"]["projection_mat"]
    image_wh_prev = cuda0["metas"]["vggt_image_wh"]
    image_wh_curr = cuda1["metas"]["vggt_image_wh"]

    with torch.no_grad():
        feat_prev_map, feat_curr_map = pool.vggt(cuda0["vggt_imgs"], cuda1["vggt_imgs"])

        q = pool.initial_embed(means_flat, rot_flat, scale_flat, opa_flat, sem_flat)
        delta_mu = torch.zeros(N, 3, device=means_flat.device, dtype=means_flat.dtype)
        delta_r = torch.zeros(N, 4, device=means_flat.device, dtype=means_flat.dtype)
        delta_r[:, 0] = 1.0

        print("=" * 70)
        print("PER-BLOCK: offset magnitude + quaternion validity")
        print("=" * 70)
        for i, block in enumerate(pool.blocks):
            offsets = block.offset_mlp(q).view(N, block.K, 2)
            attn_weights = F.softmax(block.attn_mlp(q), dim=-1)

            offset_mag = offsets.norm(dim=-1)  # (N, K) -- magnitude of each offset in [0,1]-fraction units
            offset_var_among_k = offsets.var(dim=1).sum(dim=-1)  # (N,) -- how much the K offsets differ from each other

            attn_entropy = -(attn_weights * (attn_weights + 1e-12).log()).sum(dim=-1)  # (N,)
            max_entropy = math.log(block.K)

            print(f"\n--- Block {i} ---")
            print(f"  offset magnitude (fraction of image): mean={offset_mag.mean().item():.4f}, "
                  f"max={offset_mag.max().item():.4f}  (0.05-0.15ish is a reasonable 'local "
                  f"neighborhood' range; near-0 -> not really using offsets; >0.5 -> jumping "
                  f"to unrelated parts of the image)")
            print(f"  variance among K offsets: mean={offset_var_among_k.mean().item():.6f}  "
                  f"(near-0 -> still symmetry-locked/degenerate; the whole point of the fix "
                  f"was to make this nonzero)")
            print(f"  attention entropy: mean={attn_entropy.mean().item():.4f} / max possible "
                  f"{max_entropy:.4f}  (near max -> still close to uniform/not discriminating "
                  f"among K; near 0 -> confidently picking one sample)")

            # Step forward for real to get this block's actual delta_mu/delta_r/q output
            new_delta_mu, new_delta_r, q = block(
                means_flat, delta_mu, delta_r, q, relative_transform,
                projection_mat_prev, image_wh_prev, feat_prev_map,
                projection_mat_curr, image_wh_curr, feat_curr_map,
            )
            delta_mu, delta_r = new_delta_mu, new_delta_r

            quat_norms = delta_r.norm(dim=-1)  # should be ~1.0 -- unit quaternions
            # rotation angle implied by this quaternion: angle = 2*acos(|w|)
            w = delta_r[:, 0].abs().clamp(max=1.0)
            angles_rad = 2 * torch.acos(w)
            print(f"  delta_r quaternion norm: mean={quat_norms.mean().item():.6f}, "
                  f"std={quat_norms.std().item():.2e}  (should be ~1.0 with ~0 std -- "
                  f"unit quaternions by construction)")
            print(f"  implied rotation angle (radians): mean={angles_rad.mean().item():.4f}, "
                  f"max={angles_rad.max().item():.4f}  (each block's OWN contribution should "
                  f"individually respect max_angle_rad -- check DeformHeadR's own bound)")

        print()
        print("=" * 70)
        print("FINAL (after block 4): quaternion validity + rotation magnitude")
        print("=" * 70)
        final_norms = delta_r.norm(dim=-1)
        print(f"final delta_r norm: mean={final_norms.mean().item():.6f}, "
              f"min={final_norms.min().item():.6f}, max={final_norms.max().item():.6f} "
              f"(should be ~1.0 -- Hamilton product of unit quaternions stays unit)")
        assert torch.allclose(final_norms, torch.ones_like(final_norms), atol=1e-3), (
            "final delta_r is NOT unit-norm -- a real bug in quaternion composition"
        )
        print("[PASS] final delta_r is genuinely unit-norm")

        w_final = delta_r[:, 0].abs().clamp(max=1.0)
        final_angles = 2 * torch.acos(w_final)
        print(f"final TOTAL rotation angle across all 4 blocks (radians): "
              f"mean={final_angles.mean().item():.4f}, max={final_angles.max().item():.4f} "
              f"(4 blocks composed -- can exceed a single block's max_angle_rad=0.3, that's "
              f"expected; watch for absurdly large values like >>1.2 (4*0.3), which would "
              f"suggest something is compounding incorrectly)")

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
