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


if __name__ == "__main__":
    test_quaternion_composition_hand_computed()
    print("All remaining (architecture-independent) tests passed.")
