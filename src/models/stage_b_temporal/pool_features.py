"""
src/models/stage_b_temporal/pool_features.py

Phase 5: pools CurrentFrameEncoder's multi-scale, multi-camera feature maps down to the
single (B, in_dim) vector MotionHyperNet expects (design_doc_v2.md Section 2.3's
`pool(F^3D_t)`).

Confirmed against source (EXPERIMENT_LOG.md, Phase 5 bridge investigation):
  - F^3D = F^d (x) F^c (design_doc_v2.md Section 1.5's "outer product") is never
    materialized as a literal tensor anywhere in GaussianFormer3D -- it describes what
    DeformableFeatureAggregation3D's CUDA deformable-sampling op achieves functionally.
    The actual raw material is two separate multi-scale lists: camera features
    (ms_img_feats, from img_neck's FPN) and depth features (out_dpt_multiscale, from
    pts_dpt_head) -- confirmed real shapes (scene-0061, frame 0):
      ms_img_feats:       4 levels, each (B, N=6, C=128, H_l, W_l)
      out_dpt_multiscale: 4 levels, each (B, N=6, C=112, H_l, W_l)

Design decision (logged, not left implicit): global-average-pool each level over
(N, H_l, W_l) independently, concatenate all 8 pooled vectors (4 img + 4 depth), then one
Linear layer projects down to `in_dim` (whatever MotionHyperNet expects -- 128 in the
Phase 4-validated config, matching embed_dims). Chosen over a spatial 3D-CNN alternative
given QGFusion's already-observed overfitting failure mode on this same 10-scene budget
(900-query run, train~7-8% vs val~3.92% mIoU, fixed global embeddings learning
scene-specific shortcuts) -- a coarser, lower-parameter pooling is the safer starting
point; a spatial-preserving alternative is a candidate for a later ablation, not built
blind here.

This module owns a learnable nn.Linear (the projection), so it must be included in the
same trainable parameter group as MotionHyperNet / DeformHeadMu / DeformHeadR under Stage
1 warmup (configs/stage_b_temporal.yaml) -- it is part of the new temporal module, not a
frozen Stage A submodule.
"""
import torch
import torch.nn as nn


class PoolFeatures(nn.Module):
    def __init__(self, img_channels=128, dpt_channels=112, num_levels=4, in_dim=128):
        super().__init__()
        self.num_levels = num_levels
        pooled_dim = num_levels * img_channels + num_levels * dpt_channels
        self.proj = nn.Linear(pooled_dim, in_dim)

    @staticmethod
    def _pool_one(feat_list):
        """Each feat: (B, N, C, H, W) -> pooled (B, C), averaged over (N, H, W)."""
        pooled = []
        for feat in feat_list:
            B, N, C = feat.shape[:3]
            pooled.append(feat.mean(dim=(1, 3, 4)))  # (B, C)
        return pooled

    def forward(self, ms_img_feats, out_dpt_multiscale):
        """
        Args:
            ms_img_feats: list[Tensor], len == num_levels, each (B, N, C_img, H_l, W_l).
            out_dpt_multiscale: list[Tensor], len == num_levels, each
                                 (B, N, C_dpt, H_l, W_l).
        Returns:
            (B, in_dim) tensor, ready for MotionHyperNet.
        """
        assert len(ms_img_feats) == self.num_levels, (
            f"PoolFeatures configured for {self.num_levels} levels, got "
            f"{len(ms_img_feats)} img feature levels -- check img_backbone_out_indices "
            f"hasn't changed since this module was built"
        )
        assert len(out_dpt_multiscale) == self.num_levels, (
            f"PoolFeatures configured for {self.num_levels} levels, got "
            f"{len(out_dpt_multiscale)} depth feature levels"
        )
        img_pooled = self._pool_one(ms_img_feats)      # list of (B, C_img)
        dpt_pooled = self._pool_one(out_dpt_multiscale)  # list of (B, C_dpt)
        combined = torch.cat(img_pooled + dpt_pooled, dim=-1)  # (B, pooled_dim)
        return self.proj(combined)  # (B, in_dim)