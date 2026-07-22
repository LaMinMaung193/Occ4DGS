"""
Deformation heads Phi_mu, Phi_r (design_doc_v2.md Sec 2.5) and the buffer
update rule (Sec 2.6):

    Delta_mu_t^i = Phi_mu(z_t^i)
    Delta_r_t^i  = Phi_r(z_t^i)        # small rotation quaternion, via
                                        # tanh-bounded axis-angle -> quat exp map

    mu_t^i = mu_{t-1}^i + Delta_mu_t^i
    r_t^i  = normalize( Delta_r_t^i (x) r_{t-1}^i )   # Hamilton product, NOT addition
    s_t^i, alpha_t^i, c_t^i unchanged (time-invariant)

Both heads are shared across all Gaussians, all frames, and all scenes at
inference (design_doc_v2.md Sec 2.5) -- there is exactly one Phi_mu and one
Phi_r instance for the whole model, not one per Gaussian/frame/scene.
"""

import torch
import torch.nn as nn

from .buffer import GaussianState

# ---------------------------------------------------------------------------
# Quaternion utilities. Convention: (w, x, y, z), w scalar-first, throughout.
# ---------------------------------------------------------------------------


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 (x) q2, both (..., 4), (w, x, y, z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def axis_angle_to_quat(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Quaternion exponential map. axis_angle: (..., 3), where the vector's
    direction is the rotation axis and its norm (radians) is the rotation
    angle -- this is exactly the map design_doc_v2.md Sec 2.5 calls for
    ("tanh-bounded axis-angle then quat exponential map").

    At axis_angle -> 0, this correctly limits to the identity quaternion
    (1, 0, 0, 0); the clamp_min guards only the axis normalization, not the
    angle itself, so small-but-nonzero inputs are handled smoothly.
    """
    angle = axis_angle.norm(dim=-1, keepdim=True)
    safe_angle = angle.clamp_min(1e-8)
    axis = axis_angle / safe_angle
    half = angle * 0.5
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    quat = torch.cat([w, xyz], dim=-1)
    # where the input angle was (numerically) exactly zero, axis is
    # undefined by the division above -- force identity explicitly rather
    # than relying on sin(0)=0 to zero it out, since axis itself may be NaN
    is_zero = (angle.squeeze(-1) == 0.0)
    if is_zero.any():
        identity = torch.zeros_like(quat)
        identity[..., 0] = 1.0
        quat = torch.where(is_zero.unsqueeze(-1), identity, quat)
    return quat


# ---------------------------------------------------------------------------
# Deformation heads
# ---------------------------------------------------------------------------


class DeformHeadMu(nn.Module):
    """Phi_mu: predicts per-Gaussian position delta Delta_mu_t.

    FIXED (Phase 5 real-data wiring test, EXPERIMENT_LOG.md): originally left
    unbounded on the theory that "only the rotation head bounds its output, per
    Sec 2.5" -- this was wrong in practice. An untrained head's raw delta, even
    tiny (~0.1-0.2m observed), can push a Gaussian already near the z boundary
    (z's valid window is only 6.4m, vs 80m for x/y) outside pc_range, which
    LocalAggregator's CUDA splat kernel enforces with a hard assertion (no
    clipping/masking) -- confirmed by reproducing the crash and checking each
    axis separately (combined min/max across x,y,z masked the z violation,
    since x/y's much wider range dominated the printed extremes).

    Fix: tanh-bound the raw output per axis, scaled by max_disp_xyz -- mirrors
    Stage A's own SparseGaussian3DRefinementModule restrict_xyz/unit_xyz
    pattern (same problem, same codebase, already-vetted mechanism) rather than
    inventing a different safeguard. Default max_disp_xyz=[4.0, 4.0, 1.0]
    reuses Stage A's own unit_xyz value from occ4dgs_mini_occ3d_gs6400.py verbatim
    as a starting point -- NOT re-derived for Stage B's different physical
    meaning (inter-frame motion over ~0.5s, vs. Stage A's iterative-refinement
    step size), so treat this as an explicit, revisit-worthy assumption, not a
    settled value. apply_update_rule additionally clamps to pc_range as a
    defense-in-depth backstop, in case any single per-Gaussian delta plus an
    already-near-boundary G_0 position still slips past this bound.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128,
                 max_disp_xyz=(4.0, 4.0, 1.0)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.register_buffer(
            "max_disp_xyz", torch.tensor(max_disp_xyz, dtype=torch.float32)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(z)  # (N, 3)
        return torch.tanh(raw) * self.max_disp_xyz  # (N, 3), bounded per axis


class DeformHeadR(nn.Module):
    """Phi_r: predicts a small per-Gaussian rotation as a tanh-bounded
    axis-angle vector, converted to a unit quaternion via the exponential
    map. max_angle_rad caps the per-step rotation magnitude -- purely a
    stability knob (unrated/untested value here; Phase 5 should tune this
    against real training dynamics, not treat 0.3 rad as load-bearing)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, max_angle_rad: float = 0.3):
        super().__init__()
        self.max_angle_rad = max_angle_rad
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(z)  # (N, 3)
        bounded = torch.tanh(raw) * self.max_angle_rad
        return axis_angle_to_quat(bounded)  # (N, 4), unit quaternion


# ---------------------------------------------------------------------------
# Update rule
# ---------------------------------------------------------------------------


def apply_update_rule(
    prev_state: GaussianState,
    delta_mu: torch.Tensor,
    delta_quat: torch.Tensor,
    pc_range=None,
) -> GaussianState:
    """design_doc_v2.md Sec 2.6's update rule. prev_state is read from the
    buffer (G_{t-1}); returns the new G_t. Does not itself write to the
    buffer -- callers do that explicitly (buffer.write(G_t)) so the
    read -> deform -> write steps stay visible and separately testable.

    pc_range (optional, [xmin,ymin,zmin,xmax,ymax,zmax]): if given, clamps
    new_means into this range as a defense-in-depth backstop, in case
    DeformHeadMu's own tanh bound plus an already-near-boundary prev_state
    position still pushes a Gaussian outside GaussianHead's splat kernel's
    valid volume (confirmed necessary in practice -- see DeformHeadMu's
    docstring for the real crash this guards against). Defaults to None
    (no clamp) so Phase 4's toy-sequence test, which uses synthetic
    coordinates with no relation to Occ3D's real pc_range, is unaffected."""
    new_means = prev_state.means + delta_mu
    if pc_range is not None:
        lo = new_means.new_tensor(pc_range[:3])
        hi = new_means.new_tensor(pc_range[3:])
        new_means = torch.clamp(new_means, min=lo, max=hi)
    new_rotations = quat_normalize(quat_multiply(delta_quat, prev_state.rotations))
    return GaussianState(
        means=new_means,
        rotations=new_rotations,
        scales=prev_state.scales,
        opacities=prev_state.opacities,
        semantics=prev_state.semantics,
    )
