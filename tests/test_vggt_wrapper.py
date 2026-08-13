"""
tests/test_vggt_wrapper.py

Standalone correctness test for VGGTWrapper (src/models/stage_b_temporal/
vggt_wrapper.py), using a TINY random-init Aggregator (patch_embed="conv") injected
via the test-injection constructor path -- deliberately NOT downloading the real
~1B-parameter VGGT-1B checkpoint (no HF Hub access needed to run this test; the real
checkpoint is only required for actual Gate 1+ runs on real data).

This does NOT test whether VGGT's real pretrained features are good -- only that
THIS wrapper's own code (sequence construction, layer selection, special-token
stripping, reshape, prev/curr split) is correct, independent of what checkpoint is
loaded. Matches the established pattern: shape check + a real correctness signal, not
just a shape check.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/Documents/min/GaussianFormer3D"))
# VGGT itself lives in a separate clone -- adjust this path to wherever it's cloned
# on this machine (matches GF3D_ROOT's own pattern in src/datasets/gf3d_pipeline.py).
VGGT_ROOT = os.path.expanduser("~/Documents/min/vggt")
sys.path.insert(0, VGGT_ROOT)

import torch  # noqa: E402

from vggt.models.aggregator import Aggregator  # noqa: E402
from src.models.stage_b_temporal.vggt_wrapper import VGGTWrapper  # noqa: E402


def main():
    torch.manual_seed(0)

    # Tiny, cheap config -- patch_embed="conv" makes embed_dim actually configurable
    # (unlike "dinov2_vitl14_reg", which is a fixed 1024-dim architecture regardless
    # of the embed_dim constructor arg -- confirmed by trying the default and hitting
    # a real shape-mismatch error before settling on this config for testing).
    agg = Aggregator(img_size=70, patch_size=14, embed_dim=32, depth=4, num_heads=4,
                      patch_embed="conv", num_register_tokens=4,
                      cached_layer_indices=(1, 3))
    agg.eval()

    wrapper = VGGTWrapper(aggregator=agg)
    print(f"Inferred feat_dim: {wrapper.feat_dim} (expect 2*embed_dim=64)")
    assert wrapper.feat_dim == 64

    N_cam, H, W = 3, 42, 70  # both divisible by patch_size=14
    images_prev = torch.rand(1, N_cam, 3, H, W)
    images_curr = torch.rand(1, N_cam, 3, H, W)

    with torch.no_grad():
        feat_prev, feat_curr = wrapper(images_prev, images_curr)

    H14, W14 = H // 14, W // 14
    expected_shape = (1, N_cam, wrapper.feat_dim, H14, W14)
    print(f"feat_prev shape: {tuple(feat_prev.shape)} (expect {expected_shape})")
    print(f"feat_curr shape: {tuple(feat_curr.shape)} (expect {expected_shape})")
    assert feat_prev.shape == expected_shape
    assert feat_curr.shape == expected_shape
    print("[PASS] output shapes correct: (1, N_cam, feat_dim, H', W') for both frames")

    # Correctness check: perturbing ONE specific camera's raw pixels in the PREV
    # frame should produce its LARGEST effect on that exact camera's slot in
    # feat_prev -- not a different camera, and not accidentally landing in feat_curr.
    # (Some smaller cross-influence elsewhere IS expected and correct -- VGGT's
    # global attention genuinely mixes across the whole fed-together sequence, see
    # module docstring -- so this checks the split/index-mapping is right, not that
    # frames are fully independent.)
    target_cam = 1
    images_prev_perturbed = images_prev.clone()
    images_prev_perturbed[:, target_cam] = torch.rand(1, 3, H, W) * 10.0  # big, obvious change

    with torch.no_grad():
        feat_prev_p, feat_curr_p = wrapper(images_prev_perturbed, images_curr)

    prev_deltas = [(feat_prev[:, j] - feat_prev_p[:, j]).abs().mean().item() for j in range(N_cam)]
    curr_deltas = [(feat_curr[:, j] - feat_curr_p[:, j]).abs().mean().item() for j in range(N_cam)]
    print(f"\nPerturbing PREV frame's camera {target_cam} only:")
    print(f"  feat_prev per-camera mean abs delta: {[f'{d:.4f}' for d in prev_deltas]}")
    print(f"  feat_curr per-camera mean abs delta: {[f'{d:.4f}' for d in curr_deltas]}")

    largest_idx = max(range(N_cam), key=lambda j: prev_deltas[j])
    assert largest_idx == target_cam, (
        f"largest change landed on feat_prev[{largest_idx}], expected feat_prev[{target_cam}] "
        f"-- prev/curr split or camera indexing is likely misaligned"
    )
    assert prev_deltas[target_cam] > max(curr_deltas), (
        "the perturbed camera's own feat_prev slot should show a bigger effect than "
        "ANY feat_curr slot -- if not, the prev/curr split itself may be wrong"
    )
    print(f"[PASS] largest effect correctly lands on feat_prev[{target_cam}] (the "
          f"perturbed camera's own slot), bigger than any feat_curr effect -- "
          f"prev/curr split and camera indexing are correctly aligned")

    print("\nAll VGGTWrapper checks passed.")


if __name__ == "__main__":
    main()
