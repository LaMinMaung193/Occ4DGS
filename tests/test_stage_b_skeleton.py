"""
Generic Stage B correctness checks, architecture-independent.

Originally part of the Phase 4 exit checklist for the Motion HyperNet design.
Trimmed when that architecture was replaced by the GF3D-faithful design
(professor-approved pivot) -- kept only what has no dependency on Motion
HyperNet's specific mechanism. Full original file preserved on
archive/motion-hypernet-backup and main.

TODO once GF3D-faithful's actual deformation module exists: add back a
two-frame recursion + shape test, built against the NEW module.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.stage_b_temporal import (
    GaussianState,
    ReferenceBuffer,
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
N_TOY = 64  # small toy Gaussian count, not the real N_g -- validates
            # mechanics, not real scale.
SEMANTIC_DIM = 17  # matches configs/occ4dgs_mini_occ3d_gs6400.py's semantic_dim


def make_random_gaussian_state(n=N_TOY) -> GaussianState:
    means = (
        torch.rand(n, 3) * (torch.tensor(PC_RANGE[3:]) - torch.tensor(PC_RANGE[:3]))
        + torch.tensor(PC_RANGE[:3])
    )
    raw_rot = torch.randn(n, 4)
    rotations = raw_rot / raw_rot.norm(dim=-1, keepdim=True)
    scales = torch.rand(n, 3) * 1.6 + 0.2
    opacities = torch.rand(n, 1)
    semantics = torch.randn(n, SEMANTIC_DIM)
    return GaussianState(
        means=means,
        rotations=rotations,
        scales=scales,
        opacities=opacities,
        semantics=semantics,
    )


def test_quaternion_composition_hand_computed():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    aa_90z = torch.tensor([[0.0, 0.0, math.pi / 2]])
    q_90z = axis_angle_to_quat(aa_90z)
    expected_90z = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])
    assert torch.allclose(q_90z, expected_90z, atol=1e-6), (q_90z, expected_90z)
    composed_with_identity = quat_multiply(q_90z, identity)
    assert torch.allclose(composed_with_identity, q_90z, atol=1e-6)
    composed_90_90 = quat_multiply(q_90z, q_90z)
    expected_180z = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    assert torch.allclose(composed_90_90, expected_180z, atol=1e-6), (
        composed_90_90,
        expected_180z,
    )
    q_zero = axis_angle_to_quat(torch.zeros(1, 3))
    assert torch.allclose(q_zero, identity, atol=1e-6)


def test_transform_anchor_for_projection_identity():
    from models.stage_b_temporal import transform_anchor_for_projection
    torch.manual_seed(1)
    n = 8
    mu = torch.randn(n, 3) * 10.0
    raw = torch.randn(n, 4)
    r = raw / raw.norm(dim=-1, keepdim=True)
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([3.0, -2.0, 1.0])

    mu_proj, r_proj = transform_anchor_for_projection(mu, r, pose, pose)

    assert torch.allclose(mu_proj, mu, atol=1e-5), "identity transform must not move position"
    assert torch.allclose(r_proj, r, atol=1e-5), "identity transform must not rotate"


def test_transform_anchor_for_projection_pure_translation():
    from models.stage_b_temporal import transform_anchor_for_projection
    torch.manual_seed(2)
    n = 8
    mu = torch.randn(n, 3) * 10.0
    raw = torch.randn(n, 4)
    r = raw / raw.norm(dim=-1, keepdim=True)

    pose_prev = torch.eye(4)
    pose_curr = torch.eye(4)
    translation = torch.tensor([5.0, -3.0, 2.0])
    pose_curr[:3, 3] = translation

    mu_proj, r_proj = transform_anchor_for_projection(mu, r, pose_prev, pose_curr)

    assert torch.allclose(r_proj, r, atol=1e-5), "pure translation must not rotate anchors"
    expected_mu_proj = mu - translation
    assert torch.allclose(mu_proj, expected_mu_proj, atol=1e-5), (mu_proj, expected_mu_proj)


def test_transform_anchor_for_projection_round_trip():
    from models.stage_b_temporal import transform_anchor_for_projection
    torch.manual_seed(3)
    n = 16
    mu = torch.randn(n, 3) * 10.0
    raw = torch.randn(n, 4)
    r = raw / raw.norm(dim=-1, keepdim=True)

    def random_pose(seed):
        g = torch.Generator().manual_seed(seed)
        angle = torch.rand(1, generator=g).item() * 2 * math.pi
        axis = torch.randn(3, generator=g)
        axis = axis / axis.norm()
        aa = axis * angle
        from models.stage_b_temporal import axis_angle_to_quat
        q = axis_angle_to_quat(aa.unsqueeze(0))[0]
        w, x, y, z = q.tolist()
        R = torch.tensor([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        pose = torch.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = torch.rand(3, generator=g) * 10.0 - 5.0
        return pose

    pose_prev = random_pose(100)
    pose_curr = random_pose(200)

    mu_fwd, r_fwd = transform_anchor_for_projection(mu, r, pose_prev, pose_curr)
    mu_back, r_back = transform_anchor_for_projection(mu_fwd, r_fwd, pose_curr, pose_prev)

    assert torch.allclose(mu_back, mu, atol=1e-3), \
        f"round-trip position mismatch, max diff {(mu_back - mu).abs().max().item()}"
    close_direct = torch.allclose(r_back, r, atol=1e-3)
    close_negated = torch.allclose(r_back, -r, atol=1e-3)
    assert close_direct or close_negated, \
        f"round-trip rotation mismatch, max diff {(r_back - r).abs().max().item()}"



if __name__ == "__main__":
    test_quaternion_composition_hand_computed()
    test_transform_anchor_for_projection_identity()
    test_transform_anchor_for_projection_pure_translation()
    test_transform_anchor_for_projection_round_trip()
    print("All tests passed.")


# ---------------------------------------------------------------------------
# transform_anchor_for_projection (Section 3.3, v4) -- round-trip consistency,
# not hand-derived exact values (avoids the real risk of baking in our own
# sign-convention arithmetic mistake as if it were verified ground truth).
# ---------------------------------------------------------------------------
