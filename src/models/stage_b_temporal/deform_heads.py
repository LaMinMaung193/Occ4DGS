"""
Deformation heads Phi_mu, Phi_r (design_doc_v2.md Sec 2.5), the buffer
update rule (Sec 2.6), and ego-motion compensation utilities.

    Delta_mu_t^i = Phi_mu(z_t^i)
    Delta_r_t^i  = Phi_r(z_t^i)        # small rotation quaternion, via
                                        # tanh-bounded axis-angle -> quat exp map

    mu_t^i = mu_{t-1}^i + Delta_mu_t^i
    r_t^i  = normalize( Delta_r_t^i (x) r_{t-1}^i )   # Hamilton product, NOT addition
    s_t^i, alpha_t^i, c_t^i unchanged (time-invariant), UNLESS recycled/spawned -- see
    apply_update_rule's recycle_out_of_range / spawn_offset options.

Both heads are shared across all Gaussians, all frames, and all scenes at
inference -- there is exactly one Phi_mu and one Phi_r instance for the whole model.
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
    angle.
    """
    angle = axis_angle.norm(dim=-1, keepdim=True)
    safe_angle = angle.clamp_min(1e-8)
    axis = axis_angle / safe_angle
    half = angle * 0.5
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    quat = torch.cat([w, xyz], dim=-1)
    is_zero = (angle.squeeze(-1) == 0.0)
    if is_zero.any():
        identity = torch.zeros_like(quat)
        identity[..., 0] = 1.0
        quat = torch.where(is_zero.unsqueeze(-1), identity, quat)
    return quat


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Converts a batch of 3x3 rotation matrices to unit quaternions (w, x, y, z),
    scalar-first, matching this module's convention throughout.

    R: (..., 3, 3). Returns: (..., 4). Hand-verified against identity and a
    5-degree yaw case, plus round-trip reconstruction.
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
    """
    return torch.linalg.inv(pose_curr) @ pose_prev


# ---------------------------------------------------------------------------
# Deformation heads
# ---------------------------------------------------------------------------


class DeformHeadMu(nn.Module):
    """Phi_mu: predicts per-Gaussian position delta Delta_mu_t. Tanh-bounded per
    axis, scaled by max_disp_xyz (default (4.0,4.0,1.0), reusing Stage A's own
    unit_xyz value as a starting point)."""

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
        raw = self.net(z)
        return torch.tanh(raw) * self.max_disp_xyz


class DeformHeadR(nn.Module):
    """Phi_r: predicts a small per-Gaussian rotation as a tanh-bounded
    axis-angle vector, converted to a unit quaternion via the exponential map."""

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
        raw = self.net(z)
        bounded = torch.tanh(raw) * self.max_angle_rad
        return axis_angle_to_quat(bounded)


# ---------------------------------------------------------------------------
# Update rule
# ---------------------------------------------------------------------------

RECYCLE_MARGIN_FRAC = 0.10


def _wrapped_base_position(new_means, lo, hi, below, above):
    span = hi - lo
    wrapped = new_means.clone()
    wrapped = torch.where(below, hi - RECYCLE_MARGIN_FRAC * span, wrapped)
    wrapped = torch.where(above, lo + RECYCLE_MARGIN_FRAC * span, wrapped)
    return wrapped


def apply_update_rule(
    prev_state: GaussianState,
    delta_mu: torch.Tensor,
    delta_quat: torch.Tensor,
    pc_range=None,
    recycle_out_of_range: bool = False,
    recycle_opacity: float = 0.3,
    spawn_offset: torch.Tensor = None,
    spawn_opacity: torch.Tensor = None,
    spawn_semantics: torch.Tensor = None,
) -> GaussianState:
    """design_doc_v2.md Sec 2.6's update rule. prev_state is read from the
    buffer (G_{t-1}); returns the new G_t.

    Three ways out-of-range Gaussians can be handled, in order of sophistication:
      1. Default: opacity zeroed -- honest "no data here".
      2. recycle_out_of_range=True: heuristic wraparound to a fixed margin at the
         OPPOSITE boundary, fixed recycle_opacity, semantics unchanged. Confirmed via
         direct testing to show NO improvement over option 1.
      3. spawn_offset/spawn_opacity/spawn_semantics provided (from SpawnHead): position
         = fixed wrapped base position + a LEARNED refinement offset; opacity and
         semantics are LEARNED, not fixed/inherited. Shapes (N,3)/(N,1)/(N,semantic_dim),
         unbatched, broadcasting against prev_state's batched (1,N,...) tensors.
    """
    new_means = prev_state.means + delta_mu
    new_opacities = prev_state.opacities
    new_semantics = prev_state.semantics
    if pc_range is not None:
        lo = new_means.new_tensor(pc_range[:3])
        eps = 1e-3
        hi = new_means.new_tensor(pc_range[3:]) - eps
        below = new_means < lo
        above = new_means > hi
        out_of_range = (below | above).any(dim=-1, keepdim=True)

        if spawn_offset is not None:
            wrapped_base = _wrapped_base_position(new_means, lo, hi, below, above)
            spawned_means = torch.clamp(wrapped_base + spawn_offset, min=lo, max=hi)
            clamped = torch.clamp(new_means, min=lo, max=hi)
            new_means = torch.where(out_of_range, spawned_means, clamped)
            new_opacities = torch.where(out_of_range, spawn_opacity, prev_state.opacities)
            new_semantics = torch.where(out_of_range, spawn_semantics, prev_state.semantics)
        elif recycle_out_of_range:
            wrapped = _wrapped_base_position(new_means, lo, hi, below, above)
            clamped = torch.clamp(new_means, min=lo, max=hi)
            new_means = torch.where(out_of_range, wrapped, clamped)
            recycle_val = torch.full_like(prev_state.opacities, recycle_opacity)
            new_opacities = torch.where(out_of_range, recycle_val, prev_state.opacities)
        else:
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
        semantics=new_semantics,
    )


def apply_ego_compensated_update_rule(
    prev_state: GaussianState,
    delta_mu: torch.Tensor,
    delta_quat: torch.Tensor,
    relative_transform: torch.Tensor,
    pc_range=None,
    recycle_out_of_range: bool = False,
    recycle_opacity: float = 0.3,
    spawn_offset: torch.Tensor = None,
    spawn_opacity: torch.Tensor = None,
    spawn_semantics: torch.Tensor = None,
) -> GaussianState:
    """Applies the KNOWN rigid ego-motion transform to prev_state's means/rotations
    FIRST, then applies the LEARNED residual (delta_mu, delta_quat) on top via
    apply_update_rule. relative_transform: (4,4), from compute_relative_transform.
    """
    R_ego = relative_transform[:3, :3]
    t_ego = relative_transform[:3, 3]

    means = prev_state.means
    means_flat = means.squeeze(0) if means.dim() == 3 else means
    means_ego = means_flat @ R_ego.transpose(0, 1) + t_ego
    if means.dim() == 3:
        means_ego = means_ego.unsqueeze(0)

    q_ego = rotmat_to_quat(R_ego.unsqueeze(0))
    rotations = prev_state.rotations
    rot_flat = rotations.squeeze(0) if rotations.dim() == 3 else rotations
    rot_ego = quat_normalize(quat_multiply(q_ego.expand_as(rot_flat), rot_flat))
    if rotations.dim() == 3:
        rot_ego = rot_ego.unsqueeze(0)

    ego_compensated_state = GaussianState(
        means=means_ego, rotations=rot_ego,
        scales=prev_state.scales, opacities=prev_state.opacities,
        semantics=prev_state.semantics,
    )
    return apply_update_rule(
        ego_compensated_state, delta_mu, delta_quat, pc_range=pc_range,
        recycle_out_of_range=recycle_out_of_range, recycle_opacity=recycle_opacity,
        spawn_offset=spawn_offset, spawn_opacity=spawn_opacity,
        spawn_semantics=spawn_semantics,
    )


def compute_spawn_candidate_positions(g_prev_means_flat, delta_mu, relative_transform, pc_range):
    """Computes the SAME fixed wrapped-base candidate positions apply_update_rule
    will use internally, so a caller (deform_one_step) can query the motion grid at
    these exact positions BEFORE calling apply_ego_compensated_update_rule with
    SpawnHead's outputs -- guarantees consistency.

    g_prev_means_flat: (N, 3), UNBATCHED. delta_mu: (N, 3), unbatched.
    Returns: wrapped_base (N, 3), out_of_range (N, 1) bool mask.
    """
    R_ego = relative_transform[:3, :3]
    t_ego = relative_transform[:3, 3]
    means_ego = g_prev_means_flat @ R_ego.transpose(0, 1) + t_ego
    new_means_preview = means_ego + delta_mu
    lo = new_means_preview.new_tensor(pc_range[:3])
    eps = 1e-3
    hi = new_means_preview.new_tensor(pc_range[3:]) - eps
    below = new_means_preview < lo
    above = new_means_preview > hi
    out_of_range = (below | above).any(dim=-1, keepdim=True)
    wrapped_base = _wrapped_base_position(new_means_preview, lo, hi, below, above)
    return wrapped_base, out_of_range
