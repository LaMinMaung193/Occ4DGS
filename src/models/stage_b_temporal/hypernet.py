"""
Motion HyperNet -- the one genuinely new module in Stage B
(design_doc_v2.md Sec 2.3). Predicts L compact multi-resolution 3D motion
grids {M_t^l} from a pooled current-frame feature vector:

    M_t = HyperNet(pool(F^3D_t))

Phase 4 scope note: F^3D_t (the real LiDAR-camera feature volume) doesn't
exist yet -- Stage A's encoders aren't wired into Stage B until Phase 5
(IMPLEMENTATION_ROADMAP.md). Here we take a pre-pooled feature vector of a
configurable dimension and feed it random noise in the toy-sequence test,
per the roadmap's explicit instruction to use "dummy (randomly initialized,
untrained) inputs ... feed zeros or random noise instead of real F^3D_t for
now." The module's *interface* (pooled vector in, list of dense grids out)
is what Phase 5 needs to match; the pooling operation itself belongs to
whatever encoder wiring Phase 5 adds, not to this module.
"""

from typing import List, Sequence

import torch
import torch.nn as nn


class MotionHyperNet(nn.Module):
    """
    One small MLP head per resolution level, each mapping the pooled
    current-frame feature to a flattened dense grid, reshaped to
    (B, grid_feat_dim, r, r, r).

    resolutions defaults to 3 levels (L=3, matching 4DGC/4D-GS defaults per
    design_doc_v2.md Sec 2.3). Kept deliberately small/cubic here since
    Phase 4 only needs to validate shapes and recursion, not real quality --
    revisit resolution choice once Phase 5 wires in real features and VRAM
    becomes a live constraint again (c.f. Phase 2's 2.84GB headroom finding,
    which does NOT directly transfer here since dense per-frame grids are a
    different memory profile than Stage A's sparse Gaussian set).
    """

    def __init__(
        self,
        in_dim: int,
        grid_feat_dim: int = 16,
        resolutions: Sequence[int] = (4, 8, 16),
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.grid_feat_dim = grid_feat_dim
        self.resolutions = tuple(resolutions)

        self.level_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, grid_feat_dim * (r**3)),
                )
                for r in self.resolutions
            ]
        )

    def forward(self, pooled_feat: torch.Tensor) -> List[torch.Tensor]:
        """
        pooled_feat: (B, in_dim), B is fixed to 1 in Stage B's per-clip
        processing (one sequence at a time) but kept general here.

        Returns: list of length L, each (B, grid_feat_dim, r, r, r).
        """
        if pooled_feat.dim() != 2 or pooled_feat.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected pooled_feat of shape (B, {self.in_dim}), got "
                f"{tuple(pooled_feat.shape)}"
            )
        b = pooled_feat.shape[0]
        grids = []
        for head, r in zip(self.level_heads, self.resolutions):
            flat = head(pooled_feat)  # (B, grid_feat_dim * r^3)
            grid = flat.view(b, self.grid_feat_dim, r, r, r)
            grids.append(grid)
        return grids
