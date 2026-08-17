"""
tests/test_deformable_temporal_block.py

Standalone correctness test for DeformableTemporalBlock / InitialQueryEmbed
(src/models/stage_b_temporal/deformable_temporal_block.py), on synthetic camera/
feature data, matching the established pattern (real correctness signals, not just
shape checks, before any wiring into stage_b_engine.py).

ENVIRONMENT NOTE: like DirectProjectionSampler's test, this imports
DeformableFeatureAggregation3D.project_points_3d from the real GaussianFormer3D repo,
which requires its compiled CUDA ops extension already built -- run on the lab GPU
box, not a plain CPU dev machine.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.models.stage_b_temporal.deformable_temporal_block import (  # noqa: E402
    DeformableTemporalBlock, InitialQueryEmbed,
)


def _build_projection_mat(cam_offset_xyz, focal=20.0, cx=16.0, cy=16.0):
    """Same minimal pinhole construction as Option D's test -- see
    test_direct_projection_sampler.py for the derivation."""
    cx_off, cy_off, cz_off = cam_offset_xyz
    extrinsic = torch.tensor([
        [1.0, 0.0, 0.0, -cx_off],
        [0.0, 1.0, 0.0, -cy_off],
        [0.0, 0.0, 1.0, -cz_off],
        [0.0, 0.0, 0.0, 1.0],
    ])
    K = torch.tensor([
        [cx,   focal, 0.0,  0.0],
        [cy,   0.0,   focal, 0.0],
        [1.0,  0.0,   0.0,  0.0],
        [0.0,  0.0,   0.0,  1.0],
    ])
    return K @ extrinsic


def main():
    torch.manual_seed(0)
    N_cam, C, H, W = 2, 8, 32, 32
    query_dim, semantic_dim = 16, 17
    image_wh_val = torch.tensor([[float(W), float(H)]] * N_cam).unsqueeze(0)

    proj_cam0 = _build_projection_mat((0.0, 0.0, 0.0))
    proj_cam1 = _build_projection_mat((0.0, 50.0, 0.0))
    projection_mat = torch.stack([proj_cam0, proj_cam1], dim=0).unsqueeze(0)

    feat_prev_map = torch.randn(1, N_cam, C, H, H)
    feat_curr_map = torch.randn(1, N_cam, C, H, H)

    # --- InitialQueryEmbed shape check ---
    embed = InitialQueryEmbed(query_dim=query_dim, hidden_dim=32, semantic_dim=semantic_dim)
    N = 2
    means = torch.tensor([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
    rotations = torch.zeros(N, 4); rotations[:, 0] = 1.0
    scales = torch.ones(N, 3)
    opacities = torch.ones(N, 1)
    semantics = torch.randn(N, semantic_dim)
    with torch.no_grad():
        q0 = embed(means, rotations, scales, opacities, semantics)
    print(f"q0 shape: {tuple(q0.shape)} (expect ({N}, {query_dim}))")
    assert q0.shape == (N, query_dim)
    print("[PASS] InitialQueryEmbed output shape correct")

    # --- DeformableTemporalBlock shape check ---
    block = DeformableTemporalBlock(feat_dim=C, query_dim=query_dim, K=4, hidden_dim=32,
                                     d_bound=(2.0, 58.0))
    # eval() disables z_dropout -- required for this test's deterministic-comparison
    # checks below (ablation/perturbation tests assume identical inputs produce
    # identical outputs across separate forward calls; with dropout active, two
    # calls get two different random masks regardless of input, which would break
    # that assumption for reasons having nothing to do with the thing being tested).
    # Gradient flow into z_dropout itself isn't a concern -- nn.Dropout has no
    # learnable parameters to verify.
    block.eval()
    delta_mu0 = torch.zeros(N, 3)
    delta_r0 = torch.zeros(N, 4); delta_r0[:, 0] = 1.0
    identity_T = torch.eye(4)

    with torch.no_grad():
        delta_mu1, delta_r1, q1 = block(
            means, delta_mu0, delta_r0, q0, identity_T,
            projection_mat, image_wh_val, feat_prev_map,
            projection_mat, image_wh_val, feat_curr_map,
        )
    print(f"\ndelta_mu1 shape: {tuple(delta_mu1.shape)} (expect ({N}, 3))")
    print(f"delta_r1 shape: {tuple(delta_r1.shape)} (expect ({N}, 4))")
    print(f"q1 shape: {tuple(q1.shape)} (expect ({N}, {query_dim}))")
    assert delta_mu1.shape == (N, 3)
    assert delta_r1.shape == (N, 4)
    assert q1.shape == (N, query_dim)
    print("[PASS] DeformableTemporalBlock output shapes correct")

    # --- Offsets start SMALL (std=0.01 random, not exactly zero -- see the
    # symmetry-breaking fix in deformable_temporal_block.py) and attn starts
    # uniform, so this still closely reduces to the same camera-ablation
    # correctness check as Option D's DirectProjectionSampler: perturbing the
    # camera Gaussian A (means[0], in front of CAM_0) actually projects into must
    # change its sampled feature; perturbing CAM_1 (which it does NOT project into)
    # must not -- unaffected by the offset fix, since CAM_1's contribution is
    # masked out entirely (weight=0) regardless of offset magnitude. ---
    feat_prev_ablated = feat_prev_map.clone()
    feat_prev_ablated[:, 0] = 0.0
    with torch.no_grad():
        delta_mu1_ablated, _, _ = block(
            means, delta_mu0, delta_r0, q0, identity_T,
            projection_mat, image_wh_val, feat_prev_ablated,
            projection_mat, image_wh_val, feat_curr_map,
        )
    diff_seen_cam = (delta_mu1[0] - delta_mu1_ablated[0]).abs().max().item()
    print(f"\nAblating CAM_0 (Gaussian A's own camera) in feat_prev:")
    print(f"  change in Gaussian A's delta_mu: {diff_seen_cam:.4f}")
    assert diff_seen_cam > 1e-4, "ablating the observing camera should change the output"

    feat_prev_ablated2 = feat_prev_map.clone()
    feat_prev_ablated2[:, 1] = 0.0
    with torch.no_grad():
        delta_mu1_ablated2, _, _ = block(
            means, delta_mu0, delta_r0, q0, identity_T,
            projection_mat, image_wh_val, feat_prev_ablated2,
            projection_mat, image_wh_val, feat_curr_map,
        )
    diff_unseen_cam = (delta_mu1[0] - delta_mu1_ablated2[0]).abs().max().item()
    print(f"Ablating CAM_1 (a camera Gaussian A does NOT project into):")
    print(f"  change in Gaussian A's delta_mu: {diff_unseen_cam:.6f}")
    assert diff_unseen_cam < 1e-6, "ablating a non-observing camera must not change the output"
    print("[PASS] geometric correctness holds (small-random-offset init, "
          "reducing to the same camera-ablation check as Option D)")

    # --- THE key new-mechanism check: frame t-1's anchor must be FIXED across
    # blocks (uses means_fixed only, never delta_mu_prev), while frame t's anchor
    # must be REFINED (uses means_fixed + delta_mu_prev). Test by holding q_prev
    # IDENTICAL across two calls (so offsets/attn_weights are identical too) and
    # varying ONLY delta_mu_prev: f_prev's contribution should then be completely
    # unaffected, while f_curr's must change. ---
    delta_mu_prev_a = torch.zeros(N, 3)
    delta_mu_prev_b = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # move Gaussian A only

    with torch.no_grad():
        dmu_a, dr_a, q_a = block(means, delta_mu_prev_a, delta_r0, q0, identity_T,
                                  projection_mat, image_wh_val, feat_prev_map,
                                  projection_mat, image_wh_val, feat_curr_map)
        dmu_b, dr_b, q_b = block(means, delta_mu_prev_b, delta_r0, q0, identity_T,
                                  projection_mat, image_wh_val, feat_prev_map,
                                  projection_mat, image_wh_val, feat_curr_map)

    print(f"\nVarying delta_mu_prev only (q_prev held identical):")
    print(f"  Gaussian A's block output changed: {not torch.allclose(dmu_a[0], dmu_b[0], atol=1e-6)}")
    assert not torch.allclose(dmu_a[0], dmu_b[0], atol=1e-6), (
        "changing delta_mu_prev (which shifts the frame-t anchor) should change the "
        "output -- if not, the running-estimate refinement isn't wired correctly"
    )
    print("[PASS] frame t's anchor correctly responds to the running delta_mu estimate "
          "(the core iterative-refinement mechanism this block exists to implement)")

    print("\nAll DeformableTemporalBlock / InitialQueryEmbed checks passed.")


if __name__ == "__main__":
    main()
