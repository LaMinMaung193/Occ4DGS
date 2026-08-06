"""
src/models/stage_b_temporal/spatial_pool_features.py

Step 5, piece 1 (EXPERIMENT_LOG.md): replaces PoolFeatures' full collapse-to-one-vector
with a small preserved spatial panorama, addressing the professor's feedback that
pooling destroys spatial information, AND that the 6 nuScenes cameras have real
overlapping fields of view that need explicit handling, not just uniform averaging.

Two-level design:
  Level 1 (within-camera): each camera's own feature map is pooled over height (H)
    and resampled along width (W) into K angular sub-samples, preserving spatial
    detail ACROSS that camera's own field of view, not collapsed to one scalar.
  Level 2 (cross-camera): each of the 6 cameras' K sub-samples get scattered onto a
    shared 360-degree panorama grid at their true angular position (camera center yaw
    +/- offset within its own FOV), blended smoothly in overlap zones using a
    raised-cosine window -- confirmed via tests/test_spatial_pool_features.py: genuine
    blending happens only in real overlap regions (6/24 bins with default settings),
    full 360-degree coverage with no gaps, weights sum to 1 everywhere, and ablating one
    camera's contribution measurably changes only the bins it actually overlaps into.

Camera order confirmed against src/datasets/nuscenes_mini.py's real CAM_NAMES (NOT a
simple clockwise ring -- grouped as FRONT/FRONT_RIGHT/FRONT_LEFT/BACK/BACK_RIGHT/
BACK_LEFT). Yaw angles below are NOMINAL RIG-DESIGN values, not each scene's true
calibrated extrinsics -- an approximation adopted for Gate 1 (does it run, does loss
decrease); revisit with real per-scene calibration before trusting a Gate 3/4 result.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


CAM_NAMES = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
             "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"]
NOMINAL_YAW_DEG = {
    "CAM_FRONT": 0.0, "CAM_FRONT_LEFT": 55.0, "CAM_BACK_LEFT": 110.0,
    "CAM_BACK": 180.0, "CAM_BACK_RIGHT": 250.0, "CAM_FRONT_RIGHT": 305.0,
}
ASSUMED_FOV_DEG = 80.0  # nominal, same for every camera (approximation)


def _angular_diff_deg(a, b):
    """Smallest signed difference a-b on a 360-degree circle, result in [-180,180]."""
    return (a - b + 180.0) % 360.0 - 180.0


class SpatialPoolFeatures(nn.Module):
    """
    Output: (B, out_channels, panorama_bins) -- a real, if approximate, spatial map
    around the vehicle's full 360 degrees, not a single pooled vector.
    """
    def __init__(self, img_channels=128, dpt_channels=112, num_levels=4,
                 out_channels=32, k_subsamples=4, panorama_bins=24):
        super().__init__()
        self.num_levels = num_levels
        self.k_subsamples = k_subsamples
        self.panorama_bins = panorama_bins

        cam_yaw = torch.tensor([NOMINAL_YAW_DEG[name] for name in CAM_NAMES])  # (6,)
        self.register_buffer("cam_yaw", cam_yaw)

        panorama_angles = torch.linspace(0, 360, steps=panorama_bins + 1)[:-1]  # (P,)
        self.register_buffer("panorama_angles", panorama_angles)

        # Cross-camera blend weights: (P, 6), rows sum to 1.
        diffs = _angular_diff_deg(panorama_angles[:, None], cam_yaw[None, :])  # (P,6)
        half_fov = ASSUMED_FOV_DEG / 2.0
        inside = diffs.abs() <= half_fov
        raw = torch.where(inside, 0.5 * (1 + torch.cos(math.pi * diffs / half_fov)),
                           torch.zeros_like(diffs))
        blend_weights = raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("blend_weights", blend_weights)  # (P, 6)

        # Within-camera: for each panorama bin and each camera, which of that camera's
        # K sub-samples corresponds to this bin's angle (fractional index for linear
        # interpolation along the camera's own K-sample axis). Sub-sample k covers
        # angle: cam_yaw - fov/2 + (k+0.5)*(fov/K).
        frac_idx = (diffs / ASSUMED_FOV_DEG + 0.5) * k_subsamples - 0.5  # (P,6)
        frac_idx = frac_idx.clamp(0, k_subsamples - 1)
        self.register_buffer("frac_idx", frac_idx)  # (P, 6)

        pooled_dim_per_level = img_channels + dpt_channels
        self.proj = nn.Linear(num_levels * pooled_dim_per_level, out_channels)

    def _pool_one_source(self, feat_list):
        """Each feat: (B, N=6, C, H, W) -> (B, N, C, K), collapsing H, resampling W."""
        per_level = []
        for feat in feat_list:
            B, N, C, H, W = feat.shape
            flat = feat.view(B * N, C, H, W)
            pooled_h = flat.mean(dim=2)  # (B*N, C, W) -- collapse height
            resampled = F.adaptive_avg_pool1d(pooled_h, self.k_subsamples)  # (B*N,C,K)
            per_level.append(resampled.view(B, N, C, self.k_subsamples))
        return per_level  # list of (B, N, C, K)

    def _scatter_to_panorama(self, per_level_subsamples):
        """per_level_subsamples: list of (B, N, C, K). Returns list of (B, C, P), one
        per level."""
        outs = []
        for sub in per_level_subsamples:
            B, N, C, K = sub.shape
            P = self.panorama_bins
            frac = self.frac_idx  # (P, N)
            lo = frac.floor().long().clamp(0, K - 1)  # (P, N)
            hi = (lo + 1).clamp(0, K - 1)
            w_hi = (frac - lo.float())  # (P, N)
            w_lo = 1.0 - w_hi

            sub_ = sub.permute(0, 1, 3, 2)  # (B, N, K, C)
            lo_idx = lo.t().unsqueeze(0).unsqueeze(-1).expand(B, N, P, C)  # (B,N,P,C)
            hi_idx = hi.t().unsqueeze(0).unsqueeze(-1).expand(B, N, P, C)
            sub_exp = sub_.unsqueeze(2).expand(B, N, P, K, C)  # (B,N,P,K,C)
            val_lo = torch.gather(sub_exp, 3, lo_idx.unsqueeze(3)).squeeze(3)  # (B,N,P,C)
            val_hi = torch.gather(sub_exp, 3, hi_idx.unsqueeze(3)).squeeze(3)  # (B,N,P,C)
            w_lo_ = w_lo.t().unsqueeze(0).unsqueeze(-1)  # (1,N,P,1)
            w_hi_ = w_hi.t().unsqueeze(0).unsqueeze(-1)
            interpolated = val_lo * w_lo_ + val_hi * w_hi_  # (B, N, P, C)

            blend = self.blend_weights.t().unsqueeze(0).unsqueeze(-1)  # (1, N, P, 1)
            panorama = (interpolated * blend).sum(dim=1)  # (B, P, C)
            outs.append(panorama.permute(0, 2, 1))  # (B, C, P)
        return outs

    def forward(self, ms_img_feats, out_dpt_multiscale):
        img_sub = self._pool_one_source(ms_img_feats)
        dpt_sub = self._pool_one_source(out_dpt_multiscale)
        img_pan = self._scatter_to_panorama(img_sub)
        dpt_pan = self._scatter_to_panorama(dpt_sub)
        combined = torch.cat(img_pan + dpt_pan, dim=1)  # (B, sum(C), P)
        return self.proj(combined.transpose(1, 2)).transpose(1, 2)  # (B, out_channels, P)
