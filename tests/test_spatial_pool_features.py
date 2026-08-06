"""
tests/test_spatial_pool_features.py

Step 5, piece 1 correctness test for SpatialPoolFeatures (src/models/stage_b_temporal/
spatial_pool_features.py). Not a training test -- pure shape + geometric-correctness
checks, runnable in seconds, no GPU/dataset required.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from src.models.stage_b_temporal.spatial_pool_features import SpatialPoolFeatures  # noqa: E402


def main():
    torch.manual_seed(0)
    B, N, C_img, C_dpt = 1, 6, 8, 6
    levels = [(32, 32), (16, 16), (8, 8), (4, 4)]

    ms_img_feats = [torch.randn(B, N, C_img, h, w) for h, w in levels]
    out_dpt_multiscale = [torch.randn(B, N, C_dpt, h, w) for h, w in levels]

    module = SpatialPoolFeatures(img_channels=C_img, dpt_channels=C_dpt, num_levels=4,
                                  out_channels=16, k_subsamples=4, panorama_bins=24)
    out = module(ms_img_feats, out_dpt_multiscale)
    print("Output shape:", tuple(out.shape), "(expect (1, 16, 24))")
    assert out.shape == (1, 16, 24)
    print("[PASS] output shape correct: (B, out_channels, panorama_bins)")

    # Correctness: a genuine overlap bin must actually change when a contributing
    # camera is ablated; a bin covered ONLY by a different camera must NOT change.
    with torch.no_grad():
        img_sub = module._pool_one_source(ms_img_feats)
        panorama_full = module._scatter_to_panorama(img_sub)[0]  # (B, C_img, P), level 0

        ms_img_feats_ablated = [f.clone() for f in ms_img_feats]
        ms_img_feats_ablated[0][:, 2] = 0.0  # zero out CAM_FRONT_LEFT entirely
        img_sub_ablated = module._pool_one_source(ms_img_feats_ablated)
        panorama_ablated = module._scatter_to_panorama(img_sub_ablated)[0]

        diff_at_blend_bin = (panorama_full[:, :, 2] - panorama_ablated[:, :, 2]).abs().max().item()
        diff_at_pure_front_bin = (panorama_full[:, :, 0] - panorama_ablated[:, :, 0]).abs().max().item()

    print(f"\nAblating CAM_FRONT_LEFT's contribution:")
    print(f"  change at blend bin (bin 2, ~30deg, FRONT+FRONT_LEFT overlap): {diff_at_blend_bin:.4f}")
    print(f"  change at pure-FRONT bin (bin 0, 0deg, no FRONT_LEFT contribution): {diff_at_pure_front_bin:.6f}")
    assert diff_at_blend_bin > 1e-4, "ablating a blending camera should change the blend bin's output"
    assert diff_at_pure_front_bin < 1e-6, "ablating FRONT_LEFT should NOT affect a bin purely covered by FRONT"
    print("[PASS] overlap bin genuinely blends (changes when a contributing camera is ablated); "
          "pure-FRONT bin correctly unaffected")

    print("\nAll SpatialPoolFeatures checks passed.")


if __name__ == "__main__":
    main()
