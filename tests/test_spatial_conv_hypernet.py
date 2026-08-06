"""
tests/test_spatial_conv_hypernet.py

Step 5, piece 3 correctness test for SpatialConvHyperNet
(src/models/stage_b_temporal/spatial_conv_hypernet.py). Pure shape + architectural-
capacity checks, runnable in seconds, no GPU/dataset required.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from src.models.stage_b_temporal.spatial_conv_hypernet import SpatialConvHyperNet  # noqa: E402


def main():
    torch.manual_seed(0)
    B, C_in, P = 1, 96, 24  # C_in = 3 * 32 (piece 2's concat of curr, prev, diff panoramas)
    module = SpatialConvHyperNet(in_channels=C_in, resolutions=(8, 16, 32), grid_feat_dim=4,
                                  panorama_bins=P, map2d_size=8, seed_channels=64)

    panorama = torch.randn(B, C_in, P)
    grids = module(panorama)
    for g, r in zip(grids, (8, 16, 32)):
        print(f"level r={r}: shape={g.shape}")
        assert g.shape == (B, 4, r, r, r)
    print("[PASS] all shape checks correct")

    n_params = sum(p.numel() for p in module.parameters())
    print(f"\nTotal params: {n_params:,}  (Step 4's ConvHyperNet was ~700,708 for reference)")

    # Correctness: perturbing ONE input bin should produce a NON-UNIFORM effect on the
    # output grid -- evidence spatial info can propagate through the architecture,
    # rather than being washed out into an identical effect everywhere.
    with torch.no_grad():
        panorama_zero = torch.zeros(B, C_in, P)
        panorama_spike = panorama_zero.clone()
        panorama_spike[:, :, 0] = 5.0  # spike at bin 0 only

        grids_zero = module(panorama_zero)
        grids_spike = module(panorama_spike)

        finest_diff = (grids_spike[-1] - grids_zero[-1]).abs()  # (B,4,32,32,32)
        print(f"\nFinest grid diff stats: mean={finest_diff.mean().item():.6f}  "
              f"std={finest_diff.std().item():.6f}  max={finest_diff.max().item():.6f}")
        assert finest_diff.std().item() > 1e-6, \
            "diff is perfectly uniform -- spatial info was NOT preserved at all"
        coefficient_of_variation = finest_diff.std().item() / (finest_diff.mean().item() + 1e-8)
        print(f"coefficient of variation (std/mean): {coefficient_of_variation:.4f} "
              f"(near 0 would mean uniform/non-spatial; higher means real spatial structure)")
        assert coefficient_of_variation > 0.05, "effect looks too uniform across space"

    print("[PASS] perturbing one input bin produces a genuinely NON-UNIFORM effect on "
          "the output grid -- spatial information can propagate through this "
          "architecture (untrained network -- confirms CAPACITY for spatial routing, "
          "not that it is yet semantically correct, which requires training)")

    print("\nAll SpatialConvHyperNet checks passed.")


if __name__ == "__main__":
    main()
