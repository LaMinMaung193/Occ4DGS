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


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Converts a batch of 3x3 rotation matrices to unit quaternions (w, x, y, z),
    scalar-first, matching this module's convention throughout.

    Added for Stage B ego-motion compensation (EXPERIMENT_LOG.md): measured GT motion
    showed 60-84% of voxels changing label between adjacent frames, dominated by
    ego-vehicle motion (occ_label is ego-centric per frame), not independent object
    motion. This lets the KNOWN relative ego rotation be composed onto each Gaussian's
    orientation directly, so the learned Phi_r only needs to capture the residual
    (independent object rotation), not the dominant ego-motion component.

    Uses a single-branch ("always positive w") formulation with a clamped sqrt --
    simple and numerically fine for the small-angle rotations expected here (ego yaw
    change over one ~0.5s nuScenes keyframe gap), hand-verified against known cases
    (identity, a 5-degree yaw) and a round-trip reconstruction. NOT a general-purpose
    replacement for a full branch-selecting (Shepperd's method) implementation --
    would lose precision near 180-degree rotations, which never occur in this setting.

    R: (..., 3, 3). Returns: (..., 4).
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22
    qw = torch.sqrt(torch.clamp((trace + 1.0) / 4.0, min=1e-8))
    qx = (m21 - m12) / (4.0 * qw)
    qy = (m02 - m20) / (4.0 * qw)
    qz = (m10 - m01) / (4.0 * qw)
    quat = torch.stack([qw, qx, qy, qz], dim=-1)
    return quat_normalize(quat)


def compute_relative_transform(pose_prev: torch.Tensor, pose_curr: torch.Tensor) -> torch.Tensor:
    """pose_prev, pose_curr: (4, 4) homogeneous poses mapping LOCAL coordinates (at
    that frame's own timestamp) into a shared global frame (e.g. lidar2global).
    Returns T (4, 4) such that a point in pose_prev's local frame maps into
    pose_curr's local frame: p_curr_h = T @ p_prev_h.

    Hand-verified: a static world point, transformed via this T, lands at exactly the
    position a direct world-to-local computation gives -- confirmed with a synthetic
    5m-forward + 3-degree-yaw ego motion test before this was wired into training.
    """
    return torch.linalg.inv(pose_curr) @ pose_prev


def apply_ego_compensated_update_rule(
    prev_state: GaussianState,
    delta_mu: torch.Tensor,
    delta_quat: torch.Tensor,
    relative_transform: torch.Tensor,
    pc_range=None,
) -> GaussianState:
    """Applies the KNOWN rigid ego-motion transform to prev_state's means/rotations
    FIRST (no learning involved -- exact, from the dataset's own recorded poses), then
    applies the LEARNED residual (delta_mu, delta_quat) on top via the existing
    apply_update_rule, exactly as before. The learned heads now only need to capture
    what the known ego transform does NOT explain: independent object motion plus any
    residual error -- a much smaller, more learnable signal than the combined
    ego+object motion they were previously asked to predict from scratch.

    relative_transform: (4, 4), from compute_relative_transform(pose_prev, pose_curr).
    """
    R_ego = relative_transform[:3, :3]
    t_ego = relative_transform[:3, 3]

    means = prev_state.means
    means_flat = means.squeeze(0) if means.dim() == 3 else means  # (N, 3)
    means_ego = means_flat @ R_ego.transpose(0, 1) + t_ego  # (N, 3)
    if means.dim() == 3:
        means_ego = means_ego.unsqueeze(0)

    q_ego = rotmat_to_quat(R_ego.unsqueeze(0))  # (1, 4)
    rotations = prev_state.rotations
    rot_flat = rotations.squeeze(0) if rotations.dim() == 3 else rotations  # (N, 4)
    rot_ego = quat_normalize(quat_multiply(q_ego.expand_as(rot_flat), rot_flat))
    if rotations.dim() == 3:
        rot_ego = rot_ego.unsqueeze(0)

    ego_compensated_state = GaussianState(
        means=means_ego, rotations=rot_ego,
        scales=prev_state.scales, opacities=prev_state.opacities,
        semantics=prev_state.semantics,
    )
    return apply_update_rule(ego_compensated_state, delta_mu, delta_quat, pc_range=pc_range)


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
    new_means into this range as a defense-in-depth backstop.

    FIX (EXPERIMENT_LOG.md, ego-motion compensation debugging): originally only
    clamped position, leaving opacity untouched. Fine when the only source of
    out-of-range deltas was tiny LEARNED corrections -- but ego-motion compensation
    applies a REAL rigid shift (confirmed up to 6+ meters between real held-out
    frames), routinely pushing Gaussians already near +/-40m outside the valid volume.
    Clamping only position smeared them onto the boundary wall as visible WRONG
    content (confirmed measurably worse than doing nothing via a direct do-nothing vs
    ego-compensated-zero-residual comparison). Now any Gaussian whose pre-clamp
    position falls outside pc_range also has its opacity zeroed -- honest "no data
    here" instead of a visible but geometrically wrong smear.
    """
    new_means = prev_state.means + delta_mu
    new_opacities = prev_state.opacities
    if pc_range is not None:
        lo = new_means.new_tensor(pc_range[:3])
        eps = 1e-3
        hi = new_means.new_tensor(pc_range[3:]) - eps
        out_of_range = ((new_means < lo) | (new_means > hi)).any(dim=-1, keepdim=True)
        new_means = torch.clamp(new_means, min=lo, max=hi)
        new_opacities = torch.where(
            out_of_range, torch.zeros_like(prev_state.opacities), prev_state.opacities
        )
    new_rotations = quat_normalize(quat_multiply(delta_quat, prev_state.rotations))
    return GaussianState(
        means=new_means,
        rotations=new_rotations,
        scales=prev_state.scales,
        opacities=new_opacities,
        semantics=prev_state.semantics,
    )
