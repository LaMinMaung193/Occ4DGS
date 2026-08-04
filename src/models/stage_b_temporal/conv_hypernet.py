"""
src/models/stage_b_temporal/conv_hypernet.py

Step 4 (EXPERIMENT_LOG.md, 4DGC reference review + own architecture analysis):
replaces MotionHyperNet's per-level Linear expansion -- one pooled vector fed through
a single fully-connected layer per resolution level, with ZERO spatial inductive bias
(every output cell has its own independently-learned weights) -- with a shared
transposed-convolutional decoder.

A small 3D "seed" volume is projected from the pooled feature, then progressively
upsampled via ConvTranspose3d (the standard architecture for generating
spatially-coherent volumes from a compact code, e.g. VAE/GAN decoders), tapping off a
grid at each target resolution along the way. Coarser levels' learned features
directly inform finer levels (top-down refinement) rather than each resolution being
predicted completely independently, as in the original design.

Side effect, not the primary hypothesis but worth tracking honestly: convolutional
weight-sharing is far more parameter-efficient than a dense layer, so this also
drops total parameter count substantially (~19M in Step 2's config down to <1M) --
if this version helps, some of the improvement may be attributable to reduced
overfitting risk (less capacity) rather than purely the spatial inductive bias itself.
Both effects are part of the same underlying hypothesis (structural prior instead of
brute-force memorization capacity) and are not separated in this test.
"""
from typing import List, Sequence
import torch
import torch.nn as nn


class ConvHyperNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        grid_feat_dim: int = 4,
        resolutions: Sequence[int] = (8, 16, 32),
        seed_res: int = 4,
        seed_channels: int = 64,
    ):
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.grid_feat_dim = grid_feat_dim
        self.seed_res = seed_res
        self.seed_channels = seed_channels

        expected = seed_res
        for r in self.resolutions:
            expected *= 2
            assert r == expected, (
                f"resolutions must be successive doublings from seed_res={seed_res}; "
                f"got {resolutions}, expected {expected} at this position"
            )

        self.seed_proj = nn.Linear(in_dim, seed_channels * seed_res**3)

        self.up_blocks = nn.ModuleList()
        in_ch = seed_channels
        out_channels_per_level = []
        for _ in self.resolutions:
            out_ch = max(in_ch // 2, grid_feat_dim)
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ))
            out_channels_per_level.append(out_ch)
            in_ch = out_ch

        self.level_projs = nn.ModuleList([
            nn.Conv3d(c, grid_feat_dim, kernel_size=1) for c in out_channels_per_level
        ])

    def forward(self, pooled_feat: torch.Tensor) -> List[torch.Tensor]:
        """
        pooled_feat: (B, in_dim), B fixed to 1 in Stage B's per-clip processing.
        Returns: list of length L, each (B, grid_feat_dim, r, r, r) -- same interface
        as MotionHyperNet, drop-in replaceable.
        """
        b = pooled_feat.shape[0]
        x = self.seed_proj(pooled_feat).view(
            b, self.seed_channels, self.seed_res, self.seed_res, self.seed_res
        )
        grids = []
        for up_block, level_proj in zip(self.up_blocks, self.level_projs):
            x = up_block(x)
            grids.append(level_proj(x))
        return grids
