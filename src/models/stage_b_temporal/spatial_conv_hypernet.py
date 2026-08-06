"""
src/models/stage_b_temporal/spatial_conv_hypernet.py

Step 5, piece 3 (EXPERIMENT_LOG.md): redesigns the grid generator to consume
SpatialPoolFeatures' real panorama (piece 1) plus both-frames temporal conditioning
(piece 2), instead of a single flat pooled vector.

DESIGN DECISION (Option B, chosen over Option A -- a fully polar/angle-radius-height
coordinate system matching the panorama's true geometry, kept as a possible future
path if this proves insufficient): reshape the panorama into a small 2D map via plain
interpolation, NOT a physically exact angle-to-position mapping -- an explicit,
flagged approximation, consistent with piece 1's nominal-yaw/assumed-FOV
approximations. This keeps everything downstream (Cartesian pc_range, the existing
grid query mechanism) completely unchanged, a much smaller and lower-risk change than
Option A would have been.

Reuses Step 4's exact ConvTranspose3d growth path unchanged (same output resolutions/
channels) -- only HOW the initial 3D seed is constructed changes: from real,
panorama-derived spatial content instead of from a flat vector with no spatial
structure at all.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConvHyperNet(nn.Module):
    def __init__(self, in_channels, resolutions=(8, 16, 32), grid_feat_dim=4,
                 panorama_bins=24, map2d_size=8, seed_channels=64):
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.grid_feat_dim = grid_feat_dim
        self.map2d_size = map2d_size
        self.seed_channels = seed_channels
        self.seed_res = map2d_size // 2  # matches Step 4's seed_res=4 when map2d_size=8

        # Piece 3 step 1-2: ring (panorama_bins) -> stretched to map2d_size^2 length ->
        # reshaped to a 2D map -> Conv2d projects channels, preserves spatial size.
        self.proj2d = nn.Conv2d(in_channels, seed_channels, kernel_size=3, padding=1)

        # Piece 3 step 3: downsample 2D map to seed_res x seed_res, then expand each
        # 2D position into a depth axis of size seed_res (2D -> 3D seed).
        self.downsample = nn.Conv2d(seed_channels, seed_channels, kernel_size=2, stride=2)
        self.depth_expand = nn.Conv2d(seed_channels, seed_channels * self.seed_res, kernel_size=1)

        # Piece 3 step 4: EXACT reuse of Step 4's ConvHyperNet growth path.
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

    def forward(self, panorama):
        """
        panorama: (B, C_in, P) -- e.g. piece 2's concat([curr, prev, curr-prev]),
                  P = panorama_bins (a 1D ring, NOT yet a 2D/3D spatial layout).
        Returns: list of L grids, each (B, grid_feat_dim, r, r, r) -- same interface
        as ConvHyperNet, drop-in replaceable.
        """
        B, C, P = panorama.shape
        target_len = self.map2d_size * self.map2d_size
        stretched = F.interpolate(panorama, size=target_len, mode="linear", align_corners=False)
        map2d = stretched.view(B, C, self.map2d_size, self.map2d_size)  # (B,C,8,8) approx.

        x = F.relu(self.proj2d(map2d))               # (B, seed_channels, 8, 8)
        x = F.relu(self.downsample(x))                # (B, seed_channels, 4, 4)
        x = self.depth_expand(x)                       # (B, seed_channels*4, 4, 4)
        x = x.view(B, self.seed_channels, self.seed_res, self.seed_res, self.seed_res)  # (B,C,4,4,4)

        grids = []
        for up_block, level_proj in zip(self.up_blocks, self.level_projs):
            x = up_block(x)
            grids.append(level_proj(x))
        return grids
