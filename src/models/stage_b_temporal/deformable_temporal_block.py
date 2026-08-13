"""
src/models/stage_b_temporal/deformable_temporal_block.py

VGGT_DEFORMABLE_DESIGN.md Section 5 (math) / Section 7 item 2. One refinement block
of the new VGGT-backed deformable cross-attention mechanism -- stacked 4x by the
caller (stage_b_engine.py), each with INDEPENDENT weights.

Weight-sharing decision (not stated explicitly in the design doc's math, which uses
the same symbol names at every block l=1..4): verified against the real Stage A
precedent the design doc cites ("4 iterative blocks, matching Stage A's own design")
before deciding -- GaussianOccEncoder3D (model/encoder/gaussian_encoder/
gaussian_encoder_3d.py) builds `nn.ModuleList([build(...) for op in
operation_order])`, i.e. a SEPARATE, independently-initialized module instance per
block, NOT weight-tied/shared. To faithfully match that precedent, each of the 4
stacked DeformableTemporalBlock instances (built by the caller) gets its own
independent MLP_offset/MLP_attn/UpdateQuery/feat_norm weights, AND -- following the
same logic -- its own independent DeformHeadMu/DeformHeadR instances (still the
exact same, unmodified classes from deform_heads.py -- "[UNCHANGED heads]" in
Section 5 refers to architecture, not to there being only one shared weight instance
across blocks; compare to how Stage A's own 4 `refine` layers are 4 independent
instances of the same SparseGaussian3DRefinementModule class). This is a deliberate,
disclosed DEPARTURE from deform_heads.py's own docstring note ("there is exactly one
Phi_mu and one Phi_r instance for the whole model") -- that note describes the
prior, single-shot (non-iterative) temporal path, which this new design's 4-block
structure doesn't have an equivalent of; deform_heads.py itself is unmodified, so
that invariant remains true for every OTHER path (Steps 1-4, Option D) still using
it.

K=4 deformable offset samples per anchor per camera (Liam's confirmed decision --
magnitude matches Deformable DETR/DFA3D, cheap to revisit as a single hyperparameter
if Gate 2/3 shows under-sampling). UpdateQuery is a simple residual MLP (Liam's
confirmed decision -- consistent with DeformHeadMu/DeformHeadR's own residual-MLP
shape; GRU-style gating is the specific next thing to try if the query state is shown
to be losing information across blocks, not a preemptive choice).

Implementation clarifications for two things Section 5's math states loosely,
resolved by verified precedent rather than guessed:
  - "2D pixel offsets": added directly to project_points_3d's own [0,1]-normalized
    output (`sample_pt = p_curr + offset`), so offsets must live in that same
    [0,1]-fractional space for the addition to be dimensionally meaningful -- matches
    Deformable DETR's actual convention (offsets added to NORMALIZED reference
    points, not raw pixel coordinates), not a deviation from the literature despite
    the doc's colloquial phrasing.
  - Combining across valid cameras: Section 5 writes a plain sum
    (`sum_{j valid} sum_k ...`), but DirectProjectionSampler's already-verified,
    working precedent uses a MASKED MEAN (divide by valid camera count), avoiding
    signal magnitude scaling with the scene-dependent number of valid cameras. This
    module follows that precedent for consistency, rather than the doc's literal
    unnormalized sum.

Per Section 2.1 item 3 (disclosed simplification, not a bug): MLP_offset/MLP_attn
take ONLY the query q_i^(l-1) as input -- no camera-identity signal reaches them, so
the SAME K offsets/attention-weights are applied uniformly across every camera j
(each camera's own geometry still differs via its own anchor position; only the
offset/weight VALUES are shared).

Zero-valid-camera fallback: a small learned "no-observation" embedding, same
resolution as Option D's DirectProjectionSampler (open item 5, "same recommended
resolution").

Quaternion accumulation note: Delta_r_i^(l) = quat_multiply(Delta_r_i^(l-1),
DeformHeadR(z)) is deliberately NOT re-normalized at every block, matching Section
5's math exactly -- both factors are always exactly unit-norm by construction
(identity at l=0; DeformHeadR's axis-angle exponential map always produces a unit
quaternion), and the Hamilton product of two unit quaternions is itself always
exactly unit-norm (a real mathematical property, not an approximation), so no drift
can accumulate across blocks. The single explicit normalize() in Section 5's math
happens only once, at the very end, applying the block-4 total to r_{t-1,i} -- that
final step happens in the caller (apply_update_rule), not in this file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Side effect only: puts GF3D_ROOT on sys.path (and chdir's into it) so the plain
# `from model...` import inside forward() below resolves. Matches
# direct_projection_sampler.py's own established pattern -- an omission in this
# file's first version, caught by a real ModuleNotFoundError on real-machine testing
# rather than assumed fixed.
from src.datasets.gf3d_pipeline import GF3D_ROOT  # noqa: F401
from .deform_heads import DeformHeadMu, DeformHeadR, quat_multiply


class InitialQueryEmbed(nn.Module):
    """
    q_i^(0) = Embed(mu_i, r_i, s_i, alpha_i, c_i) -- Section 5's initial query
    embedding. Computed ONCE per deformation step (not per block) -- the caller
    builds one instance of this, shared across all steps (matching how a plain
    embedding layer would normally be used, not tied to the per-block weight
    decision above, which only concerns DeformableTemporalBlock itself).
    """

    def __init__(self, query_dim=128, hidden_dim=128, semantic_dim=17):
        super().__init__()
        in_dim = 3 + 4 + 3 + 1 + semantic_dim  # means, rotations, scales, opacities, semantics
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, query_dim),
        )

    def forward(self, means, rotations, scales, opacities, semantics):
        """All args (N, *) unbatched, matching GaussianState's flattened convention
        elsewhere in this project. Returns (N, query_dim)."""
        x = torch.cat([means, rotations, scales, opacities, semantics], dim=-1)
        return self.net(x)


class DeformableTemporalBlock(nn.Module):
    """
    One iterative refinement block. z_dim = 3*feat_dim + query_dim
    ([f_prev, f_curr, f_curr-f_prev, q]).
    """

    def __init__(self, feat_dim, query_dim=128, K=4, hidden_dim=128,
                 d_bound=(2.0, 58.0), max_disp_xyz=(4.0, 4.0, 1.0), max_angle_rad=0.3):
        super().__init__()
        self.feat_dim = feat_dim
        self.query_dim = query_dim
        self.K = K
        self.d_bound = list(d_bound)
        self.z_dim = 3 * feat_dim + query_dim

        # Section 2.1 item 3: camera-agnostic -- MLP_offset/MLP_attn take only q.
        self.offset_mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, K * 2),
        )
        # Zero-init final layer -> offsets start at exactly 0 (sampling starts
        # exactly AT the anchor at init) -- matches this project's established
        # preference for safe/bounded initialization (DeformHeadMu/R's own
        # zero-motion-at-init tanh design).
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)

        self.attn_mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, K),
        )
        # Zero-init -> softmax starts exactly uniform over the K samples at init.
        nn.init.zeros_(self.attn_mlp[-1].weight)
        nn.init.zeros_(self.attn_mlp[-1].bias)

        self.no_obs_embedding = nn.Parameter(torch.zeros(feat_dim))
        nn.init.normal_(self.no_obs_embedding, std=0.02)
        self.feat_norm = nn.LayerNorm(feat_dim)  # shared prev/curr, same rationale as Option D

        self.deform_mu = DeformHeadMu(in_dim=self.z_dim, hidden_dim=hidden_dim,
                                       max_disp_xyz=max_disp_xyz)
        self.deform_r = DeformHeadR(in_dim=self.z_dim, hidden_dim=hidden_dim,
                                     max_angle_rad=max_angle_rad)

        self.query_update_mlp = nn.Sequential(
            nn.Linear(self.z_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, query_dim),
        )
        # Zero-init -> query starts unchanged at init (q^(l) = q^(l-1) + 0).
        nn.init.zeros_(self.query_update_mlp[-1].weight)
        nn.init.zeros_(self.query_update_mlp[-1].bias)

    def _sample_one_frame(self, anchor_xy, offsets, attn_weights, bev_mask, feat_map):
        """
        anchor_xy: (num_cam, N, 2), in [0,1] (project_points_3d's own convention).
        offsets: (N, K, 2), same [0,1]-fractional space.
        attn_weights: (N, K), softmax-normalized over K.
        bev_mask: (num_cam, N) bool -- validity at the ANCHOR. Individual offset
            samples that stray outside the valid image are handled by grid_sample's
            own zero-padding, not a separate per-sample mask (see module docstring).
        feat_map: (num_cam, C, H', W') -- one frame's VGGTWrapper output.

        Returns: (N, feat_dim) masked-mean-pooled, LayerNorm'd feature per Gaussian.
        """
        num_cam, N, _ = anchor_xy.shape
        device = anchor_xy.device

        sample_pts = anchor_xy.unsqueeze(2) + offsets.unsqueeze(0)  # (num_cam, N, K, 2)
        grid = 2.0 * sample_pts - 1.0  # [0,1] -> [-1,1]

        sampled = F.grid_sample(
            feat_map, grid.to(feat_map.dtype), align_corners=False, mode="bilinear",
            padding_mode="zeros",
        )  # (num_cam, C, N, K)

        weights = attn_weights.to(sampled.dtype)  # (N, K)
        weighted = (sampled * weights[None, None, :, :]).sum(dim=-1)  # (num_cam, C, N)

        mask = bev_mask.to(weighted.dtype)  # (num_cam, N)
        summed = (weighted * mask.unsqueeze(1)).sum(dim=0)  # (C, N)
        counts = mask.sum(dim=0)  # (N,)
        valid = counts > 0

        pooled = torch.zeros(self.feat_dim, N, device=device, dtype=weighted.dtype)
        if valid.any():
            pooled[:, valid] = summed[:, valid] / counts[valid].unsqueeze(0)
        pooled = pooled.transpose(0, 1)  # (N, C)
        if (~valid).any():
            pooled = pooled.clone()
            pooled[~valid] = self.no_obs_embedding.to(pooled.dtype)

        return self.feat_norm(pooled)

    def forward(self, means_fixed, delta_mu_prev, delta_r_prev, q_prev,
                relative_transform,
                projection_mat_prev, image_wh_prev, feat_prev_map,
                projection_mat_curr, image_wh_curr, feat_curr_map):
        """
        Args:
            means_fixed: (N,3), the ORIGINAL Gaussian means at t-1 -- fixed across
                all 4 blocks (Section 4: "this frame's answer never needs refining").
            delta_mu_prev, delta_r_prev: (N,3)/(N,4), running totals from the
                previous block (zero / identity quaternion for block 1).
            q_prev: (N, query_dim), running query state from the previous block.
            relative_transform: (4,4), compute_relative_transform(pose_prev,
                pose_curr) -- reused verbatim from Option D.
            projection_mat_prev/curr, image_wh_prev/curr: metas[...] for each frame,
                matching whichever feature maps are passed (VGGT's own padded
                resolution -- metas["vggt_image_wh"] -- for feat_*_map from
                VGGTWrapper).
            feat_prev_map, feat_curr_map: (1, num_cam, C, H', W') -- VGGTWrapper's
                output for this whole deformation step (computed ONCE, not per
                block -- the same feature maps are passed into all 4 blocks; only
                the anchors/query change block to block).

        Returns:
            delta_mu_new, delta_r_new, q_new -- ready to pass into the next block
            (or into apply_update_rule after block 4).
        """
        from model.encoder.gaussian_encoder.deformable_module_3d import (
            DeformableFeatureAggregation3D,
        )

        N = means_fixed.shape[0]

        # Frame t-1 anchor: FIXED across all blocks.
        key_points_prev = means_fixed.unsqueeze(0).unsqueeze(2)  # (1,N,1,3)
        points_prev, mask_prev = DeformableFeatureAggregation3D.project_points_3d(
            key_points_prev, projection_mat_prev, self.d_bound, image_wh_prev
        )
        anchor_prev = points_prev[:, 0, :, 0, :2]  # (num_cam, N, 2)
        mask_prev = mask_prev[:, 0, :, 0]  # (num_cam, N)

        # Frame t anchor: uses the RUNNING estimate, refined block by block.
        means_curr_estimate = means_fixed + delta_mu_prev
        ones = torch.ones(N, 1, device=means_fixed.device, dtype=means_fixed.dtype)
        means_h = torch.cat([means_curr_estimate, ones], dim=-1)
        means_curr_frame = (relative_transform.to(means_fixed.dtype) @ means_h.T).T[:, :3]

        key_points_curr = means_curr_frame.unsqueeze(0).unsqueeze(2)
        points_curr, mask_curr = DeformableFeatureAggregation3D.project_points_3d(
            key_points_curr, projection_mat_curr, self.d_bound, image_wh_curr
        )
        anchor_curr = points_curr[:, 0, :, 0, :2]
        mask_curr = mask_curr[:, 0, :, 0]

        offsets = self.offset_mlp(q_prev).view(N, self.K, 2)
        attn_weights = F.softmax(self.attn_mlp(q_prev), dim=-1)  # (N, K)

        f_prev = self._sample_one_frame(anchor_prev, offsets, attn_weights, mask_prev, feat_prev_map[0])
        f_curr = self._sample_one_frame(anchor_curr, offsets, attn_weights, mask_curr, feat_curr_map[0])

        z = torch.cat([f_prev, f_curr, f_curr - f_prev, q_prev], dim=-1)

        delta_mu_new = delta_mu_prev + self.deform_mu(z)
        delta_r_new = quat_multiply(delta_r_prev, self.deform_r(z))  # NOT normalized here, see module docstring
        q_new = q_prev + self.query_update_mlp(z)

        return delta_mu_new, delta_r_new, q_new


class VGGTDeformableController(nn.Module):
    """
    Ties VGGTWrapper + InitialQueryEmbed + N_BLOCKS independent DeformableTemporalBlock
    instances together into a single callable, matching this project's established
    "one module, one forward() call, ready-to-use delta_mu/delta_r" pattern (the same
    role DirectProjectionSampler plays in Option D, just internally iterative here
    instead of a single pass).
    """

    def __init__(self, vggt_wrapper, query_dim=128, K=4, num_blocks=4, hidden_dim=128,
                 d_bound=(2.0, 58.0), semantic_dim=17,
                 max_disp_xyz=(4.0, 4.0, 1.0), max_angle_rad=0.3):
        super().__init__()
        self.vggt = vggt_wrapper
        self.initial_embed = InitialQueryEmbed(query_dim=query_dim, hidden_dim=hidden_dim,
                                                semantic_dim=semantic_dim)
        self.blocks = nn.ModuleList([
            DeformableTemporalBlock(feat_dim=vggt_wrapper.feat_dim, query_dim=query_dim,
                                     K=K, hidden_dim=hidden_dim, d_bound=d_bound,
                                     max_disp_xyz=max_disp_xyz, max_angle_rad=max_angle_rad)
            for _ in range(num_blocks)
        ])

    def forward(self, means_flat, rotations_flat, scales_flat, opacities_flat,
                semantics_flat, relative_transform, cuda_prev, cuda_curr):
        """
        means_flat etc.: (N, *), unbatched -- G_{t-1}'s own fields.
        cuda_prev/cuda_curr: the SAME dicts deform_one_step already has -- reads
            cuda_*["vggt_imgs"] and cuda_*["metas"]["projection_mat"] /
            cuda_*["metas"]["vggt_image_wh"] directly. projection_mat itself is
            IDENTICAL between the ResNet and VGGT paths (padding, not resizing,
            preserves every real pixel's exact original coordinate -- Liam's
            confirmed decision); only image_wh (the padded canvas size used for
            [0,1] normalization) differs, hence the separate vggt_image_wh and no
            separate vggt_projection_mat.

        Returns: delta_mu, delta_r (N,3)/(N,4), the block-4 running totals -- ready
            for apply_update_rule / apply_ego_compensated_update_rule, the same
            contract deform_mu(z)/deform_r(z) provide in every other branch.
        """
        feat_prev_map, feat_curr_map = self.vggt(cuda_prev["vggt_imgs"], cuda_curr["vggt_imgs"])

        q = self.initial_embed(means_flat, rotations_flat, scales_flat, opacities_flat, semantics_flat)
        N = means_flat.shape[0]
        delta_mu = torch.zeros(N, 3, device=means_flat.device, dtype=means_flat.dtype)
        delta_r = torch.zeros(N, 4, device=means_flat.device, dtype=means_flat.dtype)
        delta_r[:, 0] = 1.0  # identity quaternion

        projection_mat_prev = cuda_prev["metas"]["projection_mat"]
        projection_mat_curr = cuda_curr["metas"]["projection_mat"]
        image_wh_prev = cuda_prev["metas"]["vggt_image_wh"]
        image_wh_curr = cuda_curr["metas"]["vggt_image_wh"]

        for block in self.blocks:
            delta_mu, delta_r, q = block(
                means_flat, delta_mu, delta_r, q, relative_transform,
                projection_mat_prev, image_wh_prev, feat_prev_map,
                projection_mat_curr, image_wh_curr, feat_curr_map,
            )

        return delta_mu, delta_r
