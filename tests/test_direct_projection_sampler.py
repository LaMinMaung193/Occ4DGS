"""
tests/test_direct_projection_sampler.py

Option D, piece 1 correctness test for DirectProjectionSampler (src/models/
stage_b_temporal/direct_projection_sampler.py). Shape + geometric-correctness checks
on synthetic camera/feature data, matching the established pattern (see
test_spatial_pool_features.py / test_spatial_conv_hypernet.py: a real correctness
check, not just a shape check, run standalone BEFORE any wiring into
stage_b_engine.py).

ENVIRONMENT NOTE: unlike PoolFeatures/SpatialPoolFeatures's tests, this one is NOT
CPU-only-portable -- DirectProjectionSampler imports
DeformableFeatureAggregation3D.project_points_3d from the real GaussianFormer3D repo,
and that file's module-level `from .ops import ...` requires GaussianFormer3D's
compiled CUDA extension to already be built (see direct_projection_sampler.py's
module docstring). Run this on the lab GPU box where Stage A already runs, not on a
plain CPU dev machine.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.models.stage_b_temporal.direct_projection_sampler import (  # noqa: E402
    DirectProjectionSampler,
)


def _build_projection_mat(cam_offset_xyz, focal=20.0, cx=16.0, cy=16.0):
    """
    A minimal pinhole projection matrix for a camera translated by cam_offset_xyz
    from the LiDAR origin, looking down +x (LiDAR convention), with y-right, z-down
    in its own frame -- close enough to nuScenes' real convention for a synthetic
    correctness check (this test does not need real calibration, only a *consistent*
    projection to hand-construct known-valid / known-invalid points).

    Returns a (4,4) matrix M such that for a LiDAR-frame point p=[x,y,z,1]^T,
    M @ p = [u*depth, v*depth, depth, 1]^T (project_points_3d's expected convention:
    perspective divide happens inside project_points_3d itself, so this matrix must
    NOT pre-divide).
    """
    cx_off, cy_off, cz_off = cam_offset_xyz
    # Camera-frame axes expressed in LiDAR frame: forward=+x_lidar, right=+y_lidar,
    # down=+z_lidar (so depth = x_lidar - cx_off).
    extrinsic = torch.tensor([
        [1.0, 0.0, 0.0, -cx_off],
        [0.0, 1.0, 0.0, -cy_off],
        [0.0, 0.0, 1.0, -cz_off],
        [0.0, 0.0, 0.0, 1.0],
    ])
    # Maps camera-frame (depth, right, down) -> (u*depth, v*depth, depth).
    intrinsic_row = torch.tensor([
        [0.0, focal, 0.0, cx * 1.0],   # placeholder row, overwritten below with matmul form
    ])
    # Build explicitly: depth = X_cam (forward); u = cx + focal*(Y_cam/X_cam) -> row
    # [cx, focal, 0] dotted with [X_cam, Y_cam, Z_cam] gives u*depth when we treat the
    # projection matrix as mapping [X,Y,Z,1] -> [u*depth, v*depth, depth].
    K = torch.tensor([
        [cx,   focal, 0.0,  0.0],
        [cy,   0.0,   focal, 0.0],
        [1.0,  0.0,   0.0,  0.0],
        [0.0,  0.0,   0.0,  1.0],
    ])
    return K @ extrinsic


def main():
    torch.manual_seed(0)
    N_cam, C_img, C_dpt, H, W = 2, 8, 6, 32, 32
    image_wh_val = torch.tensor([[float(W), float(H)]] * N_cam).unsqueeze(0)  # (1,N_cam,2)

    # CAM_0 sits at the origin looking down +x; CAM_1 sits far off to the side,
    # looking the same direction, so it will NOT see points near the LiDAR origin.
    proj_cam0 = _build_projection_mat((0.0, 0.0, 0.0))
    proj_cam1 = _build_projection_mat((0.0, 50.0, 0.0))
    projection_mat = torch.stack([proj_cam0, proj_cam1], dim=0).unsqueeze(0)  # (1,2,4,4)

    ms_img_feats = [torch.randn(1, N_cam, C_img, H, H) for _ in range(4)]
    out_dpt_multiscale = [torch.randn(1, N_cam, C_dpt, H, H) for _ in range(4)]

    module = DirectProjectionSampler(img_channels=C_img, dpt_channels=C_dpt, level_idx=1,
                                      d_bound=(2.0, 58.0))

    # Gaussian A: directly in front of CAM_0 (10m ahead, centered) -> valid in CAM_0,
    # invalid in CAM_1 (which is 50m away laterally, same heading).
    # Gaussian B: behind both cameras (negative depth) -> invalid in both -> should
    # fall back to the learned no-observation embedding.
    means = torch.tensor([
        [10.0, 0.0, 0.0],
        [-10.0, 0.0, 0.0],
    ])
    identity_T = torch.eye(4)

    with torch.no_grad():
        z = module(
            means, identity_T,
            ms_img_feats, out_dpt_multiscale, projection_mat, image_wh_val,
            ms_img_feats, out_dpt_multiscale, projection_mat, image_wh_val,
        )

    print("Output shape:", tuple(z.shape), f"(expect (2, {module.z_dim}))")
    assert z.shape == (2, module.z_dim)
    print("[PASS] output shape correct: (N_gaussians, 3*(C_img+C_dpt))")

    # Gaussian B (behind both cameras) should reduce to the SAME normalized fallback
    # value for both frames -- f_prev == f_curr (both are LayerNorm(no_obs_embedding),
    # deterministically identical since it's the same shared module applied to the
    # same raw input for both frames), so the diff term is exactly zero. Note: we
    # compare f_prev_b to f_curr_b, NOT to the raw module.no_obs_embedding parameter
    # -- feat_norm is applied after the fallback substitution (see forward()), so the
    # output is the normalized embedding, not the raw one.
    feat_dim = module.feat_dim
    f_prev_b = z[1, :feat_dim]
    f_curr_b = z[1, feat_dim:2 * feat_dim]
    diff_b = z[1, 2 * feat_dim:]
    assert torch.allclose(f_prev_b, f_curr_b, atol=1e-6)
    assert torch.allclose(diff_b, torch.zeros(feat_dim), atol=1e-6)
    print("[PASS] Gaussian seen by zero cameras correctly falls back to the SAME "
          "normalized no-observation value in both frames, diff term correctly zero")

    # Gaussian A should NOT match the fallback value (it's genuinely seen by CAM_0).
    f_prev_a = z[0, :feat_dim]
    assert not torch.allclose(f_prev_a, f_prev_b, atol=1e-6)
    print("[PASS] Gaussian visible to CAM_0 does not collapse to the fallback value")

    # Geometric-correctness check (matches SpatialPoolFeatures' camera-ablation
    # pattern): perturbing the ONE camera Gaussian A projects into must change its
    # sampled feature; perturbing a camera it does NOT project into must not.
    ms_img_feats_ablated = [f.clone() for f in ms_img_feats]
    ms_img_feats_ablated[1][:, 0] = 0.0  # zero CAM_0 (the one Gaussian A is seen by)
    with torch.no_grad():
        z_ablated_cam0 = module(
            means, identity_T,
            ms_img_feats_ablated, out_dpt_multiscale, projection_mat, image_wh_val,
            ms_img_feats_ablated, out_dpt_multiscale, projection_mat, image_wh_val,
        )
    diff_when_seen_cam_ablated = (z[0] - z_ablated_cam0[0]).abs().max().item()
    print(f"\nAblating CAM_0 (the camera Gaussian A actually projects into):")
    print(f"  change in Gaussian A's z: {diff_when_seen_cam_ablated:.4f}")
    assert diff_when_seen_cam_ablated > 1e-4, (
        "ablating the camera a Gaussian projects into should change its sampled feature"
    )

    ms_img_feats_ablated2 = [f.clone() for f in ms_img_feats]
    ms_img_feats_ablated2[1][:, 1] = 0.0  # zero CAM_1 (Gaussian A is NOT seen by this one)
    with torch.no_grad():
        z_ablated_cam1 = module(
            means, identity_T,
            ms_img_feats_ablated2, out_dpt_multiscale, projection_mat, image_wh_val,
            ms_img_feats_ablated2, out_dpt_multiscale, projection_mat, image_wh_val,
        )
    diff_when_unseen_cam_ablated = (z[0] - z_ablated_cam1[0]).abs().max().item()
    print(f"Ablating CAM_1 (a camera Gaussian A does NOT project into):")
    print(f"  change in Gaussian A's z: {diff_when_unseen_cam_ablated:.6f}")
    assert diff_when_unseen_cam_ablated < 1e-6, (
        "ablating a camera the Gaussian is NOT seen by must not change its sampled feature"
    )
    print("[PASS] sampled feature responds only to the camera actually observing the "
          "Gaussian -- real geometric correctness, not just a shape check")

    # Gradient check: no_obs_embedding must receive gradient for the zero-camera
    # Gaussian, and only for that path (basic sanity that the fallback is learnable).
    module.zero_grad()
    z2 = module(
        means, identity_T,
        ms_img_feats, out_dpt_multiscale, projection_mat, image_wh_val,
        ms_img_feats, out_dpt_multiscale, projection_mat, image_wh_val,
    )
    z2.sum().backward()
    assert module.no_obs_embedding.grad is not None
    assert module.no_obs_embedding.grad.abs().sum().item() > 0
    print("[PASS] no_obs_embedding receives gradient (learnable fallback, not a "
          "frozen constant)")

    print("\nAll DirectProjectionSampler checks passed.")


if __name__ == "__main__":
    main()
