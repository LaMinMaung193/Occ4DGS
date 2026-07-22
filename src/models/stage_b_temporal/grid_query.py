"""
Per-Gaussian query into the predicted motion grids (design_doc_v2.md Sec 2.4).

FLAGGED ADDITION: IMPLEMENTATION_ROADMAP.md's Phase 4 deliverables list names
exactly buffer.py, hypernet.py, deform_heads.py. This file splits the
grid-query step out as its own module rather than folding it into
deform_heads.py -- purely a code-organization choice (it's logically its own
step between the HyperNet and the deformation heads), not a scope change.
Flagging it explicitly rather than silently adding a 4th file, per this
project's established logging discipline.

FLAGGED AMBIGUITY (unresolved, needs a decision before Phase 5): design_doc_v2.md
Sec 2.4 writes

    z_t^i = concat_l[ trilinear_interp(P_{t-1}^{i,l}, M_t^l) ]

i.e. literally using the positional-encoded P (sin/cos values) as the
trilinear-interpolation query coordinate. That can't be right as written --
grid_sample-style trilinear interpolation needs a spatial coordinate in the
grid's own coordinate frame, and a vector of sin/cos values isn't one. This
is most likely notation carried over from 4DGC's paper without being
re-derived carefully for this write-up (the same class of bug as the
cam2img/img2cam mixup in EXPERIMENT_LOG.md 2026-07-16 -- copying a formula's
shape without checking what each symbol actually has to be).

This implementation instead does what 4DGC's actual mechanism (and every
other grid-feature paper, e.g. K-Planes/Instant-NGP) does: use the Gaussian's
own (normalized) mean position as the grid_sample coordinate, and treat the
positional encoding as extra per-Gaussian context, concatenated onto the
sampled grid feature rather than substituted for the coordinate. This is a
reasonable, common-pattern guess, but it IS a guess -- confirm against 4DGC's
actual source (or with Prof. Chiang) before Phase 5 makes it load-bearing.
"""

import math
from typing import List, Sequence

import torch
import torch.nn.functional as F


def normalize_means(means: torch.Tensor, pc_range: Sequence[float]) -> torch.Tensor:
    """Map world-space means (within pc_range) into [-1, 1] per axis, the
    coordinate convention torch.nn.functional.grid_sample expects."""
    pc_range_t = torch.as_tensor(pc_range, dtype=means.dtype, device=means.device)
    lo = pc_range_t[:3]
    hi = pc_range_t[3:]
    return 2.0 * (means - lo) / (hi - lo) - 1.0


def positional_encoding(norm_means: torch.Tensor, level: int) -> torch.Tensor:
    """
    4DGC Eq. 3 style positional encoding at frequency 2^level, applied to the
    already-normalized ([-1, 1]) mean position:

        P^{i,l} = [sin(2^l * pi * mu^i), cos(2^l * pi * mu^i)]

    norm_means: (N, 3), in [-1, 1]
    Returns: (N, 6)
    """
    freq = (2**level) * math.pi
    ang = norm_means * freq
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def query_motion_grid(
    means: torch.Tensor,
    grids: List[torch.Tensor],
    pc_range: Sequence[float],
) -> torch.Tensor:
    """
    For every Gaussian, trilinearly sample each grid level at its (normalized)
    mean position, and concatenate the sampled feature with a positional
    encoding computed at that level's frequency (see module docstring for the
    flagged ambiguity this resolves one way).

    means: (N, 3) world-space Gaussian means (read from the reference buffer)
    grids: list of L tensors, each (1, C, r, r, r) -- batch size fixed to 1,
           matching Stage B's per-clip processing (design_doc_v2.md Sec 2.2)
    pc_range: [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns: z_t, shape (N, L * (C + 6)) -- the per-Gaussian temporal feature
    fed to Phi_mu / Phi_r (design_doc_v2.md Sec 2.4-2.5).
    """
    n = means.shape[0]
    norm_means = normalize_means(means, pc_range)  # (N, 3), in [-1, 1]

    # grid_sample (5D case) expects:
    #   input:  (B, C, D_in, H_in, W_in)
    #   grid:   (B, D_out, H_out, W_out, 3), last dim ordered (x, y, z)
    #           matching (W, H, D)
    # We treat each Gaussian as its own singleton output location.
    sample_coords = norm_means.view(1, n, 1, 1, 3)  # (1, N, 1, 1, 3)

    per_level_feats = []
    for level, grid in enumerate(grids, start=1):
        sampled = F.grid_sample(
            grid, sample_coords, mode="bilinear", align_corners=True
        )  # (1, C, N, 1, 1)
        c = grid.shape[1]
        sampled = sampled.view(c, n).transpose(0, 1)  # (N, C)
        pe = positional_encoding(norm_means, level)  # (N, 6)
        per_level_feats.append(torch.cat([sampled, pe], dim=-1))

    return torch.cat(per_level_feats, dim=-1)  # (N, L * (C + 6))
