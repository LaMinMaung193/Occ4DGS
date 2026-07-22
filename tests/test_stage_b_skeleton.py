"""
Phase 4 exit-checklist tests (IMPLEMENTATION_ROADMAP.md):

  [ ] Buffer state after step 1 is provably G_1 (deformed), not G_0 --
      assert this directly in a unit test, don't eyeball it
  [ ] All tensor shapes match across the full chain for a 2-frame toy sequence
  [ ] Quaternion composition (Delta_r_t (x) r_{t-1}, normalized) verified
      numerically on a hand-computed example, not just "runs without error"

Dummy encoders per roadmap Phase 4 step 2: F^3D_t is replaced with random
noise (no real camera/LiDAR features exist until Phase 5). Occ3D pc_range
below is copied from configs/occ4dgs_mini_occ3d_gs6400.py for consistency,
not re-derived here.

Run directly with `python tests/test_stage_b_skeleton.py`, or with pytest.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.stage_b_temporal import (  # noqa: E402
    GaussianState,
    ReferenceBuffer,
    MotionHyperNet,
    query_motion_grid,
    DeformHeadMu,
    DeformHeadR,
    apply_update_rule,
    quat_multiply,
    axis_angle_to_quat,
)

torch.manual_seed(0)

# Occ3D pc_range, from configs/occ4dgs_mini_occ3d_gs6400.py -- kept identical
# so this toy test's normalization matches the real config's convention.
PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

N_TOY = 64  # small toy Gaussian count -- NOT the real N_g=6400; Phase 4 only
# needs to validate mechanics, not run at real scale.
SEMANTIC_DIM = 17  # matches configs/occ4dgs_mini_occ3d_gs6400.py's semantic_dim
GRID_FEAT_DIM = 16
RESOLUTIONS = (4, 8, 16)
POOLED_FEAT_DIM = 128  # stand-in for a pooled F^3D_t dimension


def make_random_gaussian_state(n=N_TOY) -> GaussianState:
    means = (
        torch.rand(n, 3) * (torch.tensor(PC_RANGE[3:]) - torch.tensor(PC_RANGE[:3]))
        + torch.tensor(PC_RANGE[:3])
    )
    raw_rot = torch.randn(n, 4)
    rotations = raw_rot / raw_rot.norm(dim=-1, keepdim=True)
    scales = torch.rand(n, 3) * 1.6 + 0.2  # roughly Phase 2's observed [0.2, 1.6] range
    opacities = torch.rand(n, 1)
    semantics = torch.randn(n, SEMANTIC_DIM)
    return GaussianState(
        means=means,
        rotations=rotations,
        scales=scales,
        opacities=opacities,
        semantics=semantics,
    )


def build_toy_modules():
    z_dim = len(RESOLUTIONS) * (GRID_FEAT_DIM + 6)  # per-level (sampled + PE), concatenated
    hypernet = MotionHyperNet(
        in_dim=POOLED_FEAT_DIM,
        grid_feat_dim=GRID_FEAT_DIM,
        resolutions=RESOLUTIONS,
    )
    phi_mu = DeformHeadMu(in_dim=z_dim)
    phi_r = DeformHeadR(in_dim=z_dim)
    return hypernet, phi_mu, phi_r


def deform_one_frame(prev_state, hypernet, phi_mu, phi_r):
    """One Stage B step: buffer-read -> hypernet -> grid query -> heads ->
    update rule. Does not write to the buffer -- caller does that."""
    dummy_f3d_pooled = torch.randn(1, POOLED_FEAT_DIM)  # Phase 4: dummy encoder output
    grids = hypernet(dummy_f3d_pooled)
    z = query_motion_grid(prev_state.means, grids, PC_RANGE)
    delta_mu = phi_mu(z)
    delta_r = phi_r(z)
    new_state = apply_update_rule(prev_state, delta_mu, delta_r)
    return new_state, grids, z, delta_mu, delta_r


# ---------------------------------------------------------------------------
# Exit checklist item 3: quaternion composition, hand-computed
# ---------------------------------------------------------------------------


def test_quaternion_composition_hand_computed():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    # 90 degree rotation about +z: axis-angle = (0, 0, pi/2)
    aa_90z = torch.tensor([[0.0, 0.0, math.pi / 2]])
    q_90z = axis_angle_to_quat(aa_90z)
    expected_90z = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])
    assert torch.allclose(q_90z, expected_90z, atol=1e-6), (q_90z, expected_90z)

    # composing with identity should return the same rotation unchanged
    composed_with_identity = quat_multiply(q_90z, identity)
    assert torch.allclose(composed_with_identity, q_90z, atol=1e-6)

    # two consecutive 90 degree z-rotations should compose to a 180 degree
    # z-rotation: axis-angle (0, 0, pi) -> quat (cos(pi/2), 0, 0, sin(pi/2))
    #           = (0, 0, 0, 1)
    composed_90_90 = quat_multiply(q_90z, q_90z)
    expected_180z = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    assert torch.allclose(composed_90_90, expected_180z, atol=1e-6), (
        composed_90_90,
        expected_180z,
    )

    # zero rotation must map to identity exactly (guards the axis_angle=0
    # division-by-zero path in axis_angle_to_quat)
    q_zero = axis_angle_to_quat(torch.zeros(1, 3))
    assert torch.allclose(q_zero, identity, atol=1e-6)


# ---------------------------------------------------------------------------
# grid_sample coordinate-convention sanity check (catches an img2cam-style
# axis-order bug early, before it's buried under real features in Phase 5)
# ---------------------------------------------------------------------------


def test_grid_sample_coordinate_convention():
    from models.stage_b_temporal.grid_query import query_motion_grid

    # a single-level, 2x2x2 grid with a distinct value at each corner voxel
    r = 2
    grid = torch.zeros(1, 1, r, r, r)
    val = 0
    for d in range(r):
        for h in range(r):
            for w in range(r):
                grid[0, 0, d, h, w] = val
                val += 1

    # query the 8 corners of the [-1, 1]^3 cube (align_corners=True maps
    # them exactly onto the 8 grid voxel centers)
    corners_norm = torch.tensor(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    # convert normalized [-1, 1] coords back to a fake pc_range of [-1, 1]
    # (identity mapping) so query_motion_grid's normalize_means is a no-op
    pc_range = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]

    z = query_motion_grid(corners_norm, [grid], pc_range)
    sampled_vals = z[:, 0]  # first channel of the single grid level, before PE

    # grid_sample's grid[..., :] order is (x, y, z) matching dims (W, H, D);
    # our indexing loop above filled the grid as grid[d, h, w] = val with
    # (x, y, z) <-> (w, h, d), i.e. x fastest, mirroring the loop nesting.
    expected = torch.tensor(
        [
            grid[0, 0, d, h, w].item()
            for x in (0, 1)
            for y in (0, 1)
            for z in (0, 1)
            for d, h, w in [(z, y, x)]
        ]
    )
    assert torch.allclose(sampled_vals, expected, atol=1e-5), (sampled_vals, expected)


# ---------------------------------------------------------------------------
# Exit checklist items 1 & 2: recursion + shapes, full 2-frame toy sequence
# ---------------------------------------------------------------------------


def test_two_frame_toy_sequence_shapes_and_recursion():
    hypernet, phi_mu, phi_r = build_toy_modules()

    g0 = make_random_gaussian_state()
    buffer = ReferenceBuffer(g0)
    assert buffer.write_count == 0

    # sanity: buffer clones on init, so its internal state is not the same
    # tensor object as the caller's g0 (guards accidental aliasing bugs)
    assert buffer.read().means is not g0.means
    assert torch.allclose(buffer.read().means, g0.means)

    # ---- frame 1 ----
    prev = buffer.read()  # should be G_0
    assert torch.allclose(prev.means, g0.means)

    g1, grids_1, z_1, delta_mu_1, delta_r_1 = deform_one_frame(
        prev, hypernet, phi_mu, phi_r
    )

    # shape checks across the full chain
    assert len(grids_1) == len(RESOLUTIONS)
    for grid, r in zip(grids_1, RESOLUTIONS):
        assert grid.shape == (1, GRID_FEAT_DIM, r, r, r), grid.shape
    assert z_1.shape == (N_TOY, len(RESOLUTIONS) * (GRID_FEAT_DIM + 6)), z_1.shape
    assert delta_mu_1.shape == (N_TOY, 3), delta_mu_1.shape
    assert delta_r_1.shape == (N_TOY, 4), delta_r_1.shape
    assert g1.means.shape == (N_TOY, 3)
    assert g1.rotations.shape == (N_TOY, 4)
    assert g1.scales.shape == (N_TOY, 3)
    assert g1.opacities.shape == (N_TOY, 1)
    assert g1.semantics.shape == (N_TOY, SEMANTIC_DIM)

    # rotations must stay unit-norm after composition
    rot_norms_1 = g1.rotations.norm(dim=-1)
    assert torch.allclose(rot_norms_1, torch.ones(N_TOY), atol=1e-5)

    # time-invariant fields must be untouched, not just "same shape"
    assert torch.equal(g1.scales, g0.scales)
    assert torch.equal(g1.opacities, g0.opacities)
    assert torch.equal(g1.semantics, g0.semantics)

    buffer.write(g1)
    assert buffer.write_count == 1

    # *** the actual exit-checklist assertion: buffer now holds G_1, not G_0 ***
    held_after_step1 = buffer.read()
    assert not torch.allclose(held_after_step1.means, g0.means), (
        "buffer still holds G_0's means after write(g1) -- recursion is "
        "silently re-reading G_0 instead of advancing"
    )
    assert torch.allclose(held_after_step1.means, g1.means)
    assert torch.allclose(held_after_step1.rotations, g1.rotations)
    # aliasing guard: the buffer's write() must clone, not store g1 by reference
    assert held_after_step1.means is not g1.means

    # ---- frame 2 ----
    prev2 = buffer.read()  # must be G_1, confirmed above -- NOT G_0
    assert not torch.allclose(prev2.means, g0.means)

    g2, grids_2, z_2, delta_mu_2, delta_r_2 = deform_one_frame(
        prev2, hypernet, phi_mu, phi_r
    )

    assert g2.means.shape == (N_TOY, 3)
    assert g2.rotations.shape == (N_TOY, 4)
    rot_norms_2 = g2.rotations.norm(dim=-1)
    assert torch.allclose(rot_norms_2, torch.ones(N_TOY), atol=1e-5)

    buffer.write(g2)
    assert buffer.write_count == 2

    held_after_step2 = buffer.read()
    assert not torch.allclose(held_after_step2.means, g1.means), (
        "buffer still holds G_1's means after write(g2) -- recursion did "
        "not advance to G_2"
    )
    assert not torch.allclose(held_after_step2.means, g0.means)
    assert torch.allclose(held_after_step2.means, g2.means)


if __name__ == "__main__":
    test_quaternion_composition_hand_computed()
    print("[PASS] quaternion composition (hand-computed)")

    test_grid_sample_coordinate_convention()
    print("[PASS] grid_sample coordinate convention")

    test_two_frame_toy_sequence_shapes_and_recursion()
    print("[PASS] 2-frame toy sequence: shapes + recursion (buffer provably G_1/G_2, not G_0)")

    print("\nAll Phase 4 exit-checklist tests passed.")
