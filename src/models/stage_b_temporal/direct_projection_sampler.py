"""
src/models/stage_b_temporal/direct_projection_sampler.py

Option D (docs/OPTION_D_DESIGN.md): skips the intermediate 3D motion grid entirely.
For each Gaussian, projects its position directly into both the previous and current
frame's real camera images, bilinearly samples real image features at both locations,
and returns z = concat([f_prev, f_curr, f_curr - f_prev]) -- ready to feed straight
into the existing, UNCHANGED DeformHeadMu/DeformHeadR.

Reuse, not reimplementation (verified line-by-line against the real GaussianFormer3D
source, model/encoder/gaussian_encoder/deformable_module_3d.py, before this file was
written):
  - key_points: (bs, num_anchor, num_pts, 3), in LiDAR coordinates -- confirmed by the
    function's own inline comment.
  - projection_mat: (bs, num_cam, 4, 4), a single combined intrinsics+extrinsics
    matrix -- confirmed via the matmul on homogeneous [x,y,z,1].
  - Perspective divide by depth (clamped >eps), THEN normalized to [0, 1] (fraction of
    image width/height) via a separate divide by image_wh -- NOT [-1, 1]. grid_sample
    needs [-1, 1], so this module does the `2*x - 1` rescale itself, after calling
    project_points_3d, before sampling.
  - Depth channel normalized via d_bound: (depth - d_bound[0]) / (d_bound[1] -
    d_bound[0]). Only d_bound[0]/d_bound[1] are used by project_points_3d; d_bound[2]
    (a step size used elsewhere) is irrelevant here.
  - bev_mask: True where depth > eps (in front of camera) AND both normalized
    coordinates in (0, 1) (inside image bounds).
  - Output shapes: points_3d -> (num_cam, bs, num_anchor, num_pts, 3);
    bev_mask -> (num_cam, bs, num_anchor, num_pts).
d_bound=(2.0, 58.0) matches configs/occ4dgs_mini_occ3d_gs6400.py's real d_bound[:2]
exactly (line 28: d_bound = [2.0, 58, 0.5]) -- not invented.

IMPORTANT ENVIRONMENT NOTE (discovered while verifying this file against real source,
not anticipated by OPTION_D_DESIGN.md): project_points_3d is a @staticmethod on
DeformableFeatureAggregation3D, but the containing file
(deformable_module_3d.py) does `from .ops import ...` at MODULE level, and `.ops`
wraps a compiled CUDA extension (ops/src/deformable_aggregation_cuda.cu, built via
ops/setup.py). Merely importing this file -- even just to reach the pure-PyTorch
project_points_3d staticmethod -- requires that extension to already be built and
importable. This is not a NEW dependency (Stage A/CurrentFrameEncoder already load
the full segmentor, which pulls this in), but it does mean this module cannot be
unit-tested on a CPU-only / no-compiled-ops machine -- only on the lab GPU box where
Stage A already runs. Flagging explicitly rather than silently assuming portability.

Design decisions taken per OPTION_D_DESIGN.md Section 4 (open decisions), values
chosen as the "simplest starting choice" the doc names for each:
  1. Feature level: ONE mid-resolution level (default level_idx=1 of 4), not multiple
     concatenated levels -- get Gate 1/2 working first.
  2. Zero-valid-camera fallback: a single learnable "no-observation" embedding vector,
     broadcast per-frame when a Gaussian's masked mean has zero valid cameras.
  3. Both ms_img_feats AND out_dpt_multiscale sampled and concatenated (matches how
     PoolFeatures/SpatialPoolFeatures already combine both sources), not RGB features
     alone.
  4. Masked mean (not a learned re-weighting / "Option C") -- Gate 1/2 baseline only.

NORMALIZATION VARIANT (EXPERIMENT_LOG.md 2026-08-08-option-d-direct-projection-gate1-
gate2-gate3, "next planned step"): the first Gate 1-3 pass (this variant's predecessor)
showed a slow Gate 2 climb (never crossed positive until the step budget was doubled)
and a Gate 3 trough that oscillated repeatedly into -0.84 to -1.05 territory, unlike
every prior architecture's single-collapse-then-plateau shape. Leading hypothesis:
every prior architecture's z came from query_motion_grid_pe_coordinate's bounded
[-1,1] sin/cos positional encoding; this module's z was raw, unnormalized, high-
dimensional (720-d) concatenated CNN feature-map values, with no normalization
anywhere before DeformHeadMu/DeformHeadR. Added a single nn.LayerNorm(feat_dim),
SHARED across both frames and applied to each frame's pooled feature independently
(not one LayerNorm over the whole concatenated z -- see feat_norm's own docstring in
__init__ for why: a single joint LayerNorm has a separate learnable affine per
position, which would break the "identical fallback in both frames -> exactly zero
diff" invariant). This is a NEW variant with its own fresh Gate 1->2->3 pass, not a
continuation of the un-normalized run's numbers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Side effect only: puts GF3D_ROOT on sys.path (and chdir's into it) so the plain
# `from model...` import below resolves. Matches src/training/stage_b_engine.py's own
# established ordering convention ("Must come BEFORE ... GaussianFormer3D's own
# top-level modules"). Not re-derived here -- reused verbatim.
from src.datasets.gf3d_pipeline import GF3D_ROOT  # noqa: F401


class DirectProjectionSampler(nn.Module):
    """
    Per-Gaussian temporal conditioning feature, built by projecting each Gaussian
    directly into real camera images at both t-1 and t, rather than via an
    intermediate learned 3D motion grid (OPTION_D_DESIGN.md Section 2).

    z_dim = 3 * (img_channels + dpt_channels)  # [f_prev, f_curr, f_curr - f_prev]
    """

    def __init__(self, img_channels=128, dpt_channels=112, level_idx=1,
                 d_bound=(2.0, 58.0)):
        super().__init__()
        self.level_idx = level_idx
        self.d_bound = list(d_bound)
        self.feat_dim = img_channels + dpt_channels
        self.z_dim = 3 * self.feat_dim

        # Section 4, decision 2: learned fallback for Gaussians with zero valid
        # cameras at a given frame (e.g. directly below the vehicle). Small random
        # init (not exact zero) so it's a real learnable parameter from step one,
        # not dead weight the optimizer has to discover from a zero start.
        self.no_obs_embedding = nn.Parameter(torch.zeros(self.feat_dim))
        nn.init.normal_(self.no_obs_embedding, std=0.02)

        # Normalization variant (see module docstring): every prior architecture's z
        # was bounded to [-1,1] by construction (query_motion_grid_pe_coordinate's
        # sin/cos encoding). This z is raw, unnormalized, concatenated CNN features
        # with no bound at all -- LayerNorm gives DeformHeadMu/DeformHeadR a
        # consistently-scaled input, directly targeting the asymmetry flagged as the
        # leading hypothesis for the un-normalized variant's slow Gate 2 convergence
        # and Gate 3 trough instability.
        #
        # Applied per-frame (feat_dim, shared weights) rather than once over the full
        # concatenated z_dim, and the diff is computed from the NORMALIZED values --
        # deliberately, not incidentally: a single LayerNorm over the whole
        # concatenated vector has a separate learnable affine (weight/bias) per
        # position, so two IDENTICAL raw per-frame features (e.g. the no-observation
        # fallback, which reuses the same embedding for both frames) would pick up
        # DIFFERENT affine parameters depending on which block they landed in and no
        # longer match after normalization -- corrupting the "no real signal -> zero
        # diff" property. Sharing one LayerNorm(feat_dim) module across both frames
        # and differencing afterward keeps that invariant exactly (same module, same
        # input -> same output, deterministically, so equal raw features stay equal).
        self.feat_norm = nn.LayerNorm(self.feat_dim)

    def _sample_one_frame(self, means, img_feat_level, dpt_feat_level,
                           projection_mat, image_wh):
        """
        Args:
            means: (N, 3), Gaussian positions in the LiDAR frame `projection_mat`
                expects (i.e. already transformed into that frame by the caller).
            img_feat_level: (1, N_cam, C_img, H, W) -- ms_img_feats[self.level_idx].
            dpt_feat_level: (1, N_cam, C_dpt, H, W) -- out_dpt_multiscale[self.level_idx].
            projection_mat: (1, N_cam, 4, 4) -- metas["projection_mat"].
            image_wh: (1, N_cam, 2) -- metas["image_wh"].

        Returns:
            (N, C_img + C_dpt) masked-mean-pooled feature per Gaussian, with the
            learned no-observation embedding substituted for any Gaussian seen by
            zero cameras.
        """
        # Imported here (not at module top) so a bare `import` of this file doesn't
        # require the compiled CUDA ops chain unless a frame is actually sampled --
        # see the module-level docstring's environment note. This does NOT change
        # what's reused: still the real, unmodified staticmethod, called verbatim.
        from model.encoder.gaussian_encoder.deformable_module_3d import (
            DeformableFeatureAggregation3D,
        )

        device = means.device
        N = means.shape[0]

        # (bs=1, num_anchor=N, num_pts=1, 3) -- project_points_3d's expected shape.
        key_points = means.unsqueeze(0).unsqueeze(2)
        points_3d, bev_mask = DeformableFeatureAggregation3D.project_points_3d(
            key_points, projection_mat, self.d_bound, image_wh
        )
        # points_3d: (num_cam, bs=1, N, num_pts=1, 3), (x,y) in [0,1], depth-normed z.
        # bev_mask:  (num_cam, bs=1, N, num_pts=1).
        points_3d = points_3d[:, 0, :, 0, :]  # (num_cam, N, 3)
        bev_mask = bev_mask[:, 0, :, 0]       # (num_cam, N)

        # Rescale [0,1] -> [-1,1] for grid_sample (see module docstring -- this
        # project's own normalization convention differs from grid_sample's).
        grid = 2.0 * points_3d[..., :2] - 1.0          # (num_cam, N, 2)
        grid = grid.unsqueeze(2)                        # (num_cam, N, 1, 2)

        img_feat = img_feat_level[0]  # (N_cam, C_img, H, W)
        dpt_feat = dpt_feat_level[0]  # (N_cam, C_dpt, H, W)
        assert img_feat.shape[0] == grid.shape[0] == dpt_feat.shape[0], (
            f"DirectProjectionSampler: camera-count mismatch -- img_feat has "
            f"{img_feat.shape[0]}, dpt_feat has {dpt_feat.shape[0]}, projection_mat "
            f"implies {grid.shape[0]} -- these must all agree (N_cam=6 throughout "
            f"this project)."
        )

        sampled_img = F.grid_sample(
            img_feat, grid.to(img_feat.dtype), align_corners=False, mode="bilinear",
            padding_mode="zeros",
        )  # (N_cam, C_img, N, 1)
        sampled_dpt = F.grid_sample(
            dpt_feat, grid.to(dpt_feat.dtype), align_corners=False, mode="bilinear",
            padding_mode="zeros",
        )  # (N_cam, C_dpt, N, 1)
        sampled = torch.cat([sampled_img, sampled_dpt], dim=1).squeeze(-1)  # (N_cam, C, N)

        mask = bev_mask.to(sampled.dtype)               # (num_cam, N)
        summed = (sampled * mask.unsqueeze(1)).sum(dim=0)   # (C, N)
        counts = mask.sum(dim=0)                             # (N,)
        valid = counts > 0

        pooled = torch.zeros(self.feat_dim, N, device=device, dtype=sampled.dtype)
        if valid.any():
            pooled[:, valid] = summed[:, valid] / counts[valid].unsqueeze(0)
        pooled = pooled.transpose(0, 1)  # (N, C)

        if (~valid).any():
            pooled = pooled.clone()
            pooled[~valid] = self.no_obs_embedding.to(pooled.dtype)

        return pooled  # (N, feat_dim)

    def forward(self, means, relative_transform,
                ms_img_feats_prev, out_dpt_multiscale_prev, projection_mat_prev, image_wh_prev,
                ms_img_feats_curr, out_dpt_multiscale_curr, projection_mat_curr, image_wh_curr):
        """
        Args:
            means: (N, 3), Gaussian means in G_{t-1}'s LiDAR frame (no transform
                needed for the "prev" projection -- already the right frame).
            relative_transform: (4, 4), T such that p_curr_h = T @ p_prev_h --
                pass compute_relative_transform(pose_prev, pose_curr)'s output
                straight through, unchanged (already built/tested this project).
            ms_img_feats_*, out_dpt_multiscale_*: lists of per-level tensors as
                returned by CurrentFrameEncoder.encode -- only [self.level_idx] is
                used from each.
            projection_mat_*, image_wh_*: metas["projection_mat"] / metas["image_wh"]
                for each frame.

        Returns:
            z: (N, self.z_dim) == (N, 3*(C_img+C_dpt)) --
               concat([f_prev, f_curr, f_curr - f_prev]) with f_prev/f_curr each
               passed through the same shared LayerNorm(feat_dim) first (see
               feat_norm's docstring in __init__ for why per-frame + shared, not one
               LayerNorm over the whole concatenated vector) -- ready for
               DeformHeadMu/DeformHeadR unchanged (only their in_dim changes).
        """
        f_prev = self.feat_norm(self._sample_one_frame(
            means,
            ms_img_feats_prev[self.level_idx], out_dpt_multiscale_prev[self.level_idx],
            projection_mat_prev, image_wh_prev,
        ))

        ones = torch.ones(means.shape[0], 1, device=means.device, dtype=means.dtype)
        means_h = torch.cat([means, ones], dim=-1)  # (N, 4)
        means_curr_frame = (relative_transform.to(means.dtype) @ means_h.T).T[:, :3]

        f_curr = self.feat_norm(self._sample_one_frame(
            means_curr_frame,
            ms_img_feats_curr[self.level_idx], out_dpt_multiscale_curr[self.level_idx],
            projection_mat_curr, image_wh_curr,
        ))

        return torch.cat([f_prev, f_curr, f_curr - f_prev], dim=-1)
