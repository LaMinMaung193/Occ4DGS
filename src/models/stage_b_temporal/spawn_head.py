"""
src/models/stage_b_temporal/spawn_head.py

Learned recycling for Gaussians pushed out of pc_range by ego-motion compensation
(EXPERIMENT_LOG.md). The first attempt (heuristic random-wraparound + fixed opacity,
inherited/stale semantics) showed no measurable improvement over plain
opacity-zeroing -- placement carried no information about where real new content
actually is, and inherited semantics were very likely wrong for the new location.

This replaces that heuristic with a small learned head, conditioned on:
  (a) the motion grid's own feature at the candidate (wrapped) position -- the SAME
      grids MotionHyperNet already predicts, so no new global feature source needed;
  (b) the pooled scene-level feature (global motion context).
predicting, for each recycled Gaussian: a small position REFINEMENT on top of a fixed,
deterministic base position (not random -- see deform_heads.py's apply_update_rule
docstring for why determinism matters here), a learned opacity, and learned semantics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpawnHead(nn.Module):
    def __init__(self, grid_feat_dim, pooled_dim, hidden_dim=128, semantic_dim=17,
                 max_offset=2.0):
        super().__init__()
        in_dim = grid_feat_dim + pooled_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Linear(hidden_dim, 3)
        self.opacity_head = nn.Linear(hidden_dim, 1)
        self.semantics_head = nn.Linear(hidden_dim, semantic_dim)
        self.max_offset = max_offset

    def forward(self, z_candidate: torch.Tensor, pooled_global: torch.Tensor):
        """
        z_candidate: (N, grid_feat_dim) -- motion grid features queried at each
            recycled Gaussian's fixed candidate position.
        pooled_global: (1, pooled_dim) -- broadcast to all N recycled Gaussians.
        Returns: offset (N,3) tanh-bounded to +/-max_offset, opacity (N,1) in [0,1]
            (sigmoid), semantics (N,semantic_dim) via softplus.
        """
        pooled_expanded = pooled_global.expand(z_candidate.shape[0], -1)
        x = torch.cat([z_candidate, pooled_expanded], dim=-1)
        h = self.trunk(x)
        offset = torch.tanh(self.offset_head(h)) * self.max_offset
        opacity = torch.sigmoid(self.opacity_head(h))
        semantics = F.softplus(self.semantics_head(h))
        return offset, opacity, semantics
