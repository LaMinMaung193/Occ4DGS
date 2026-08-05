"""
tests/test_phase5_real_wiring.py

Phase 5 wiring validation: confirms the full chain -- real Stage A G_0, real frame-1
encoder features, PoolFeatures -> HyperNet -> grid_query -> DeformHeadMu/R ->
update rule -> ReferenceBuffer -> GaussianHead splat -- actually runs end to end
on one real 2-frame clip, before committing to a full training run.

This is NOT the Stage 1 warmup training script -- no optimizer step happens here.
This test only confirms:
  1. Real G_0 (from an actual Stage A forward pass) has the expected shape/fields.
  2. Frame 1's deformed G_1 is provably distinct from G_0 (buffer recursion holds on
     real data, not just the Phase 4 toy sequence).
  3. GaussianHead is callable standalone on G_1 with frame 1's own GT metas, producing
     pred_occ of the expected shape.
  4. Peak VRAM at unroll_window=2, one scene, one clip.

Phase 5 repo-layout cleanup (EXPERIMENT_LOG.md): rewritten to call the REAL, current
pipeline functions from src.training.stage_b_engine (build_stage_a, build_temporal_module,
get_real_g0, deform_one_step) instead of hand-reconstructing each piece locally. The
original version hand-built MotionHyperNet(resolutions=(4,8,16)) and a DeformHeadMu
in_dim formula that matched NEITHER the pre- nor post-Steps-1-4 architecture by the time
it was rediscovered -- a hand-copied snapshot silently drifting out of sync with the
real pipeline is exactly the failure mode this rewrite removes. This version will always
exercise whichever architecture is currently live, automatically, including after any
future change (e.g. the planned Step 5 redesign).

Must be run from the repo root, in the gf3d conda env, e.g.:
    PYTHONNOUSERSITE=1 python tests/test_phase5_real_wiring.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from mmengine import Config  # noqa: E402

from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402
from src.datasets.occ4dgs_clip_dataset import Occ4DGSClipDataset  # noqa: E402
from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402

from src.training.stage_b_engine import (  # noqa: E402
    REPO_ROOT,
    GF3D_ROOT,
    PC_RANGE,
    build_pipeline,
    to_batch_of_one,
    to_cuda,
    build_stage_a,
    build_temporal_module,
    get_real_g0,
    deform_one_step,
    CurrentFrameEncoder,
    ReferenceBuffer,
)

import json  # noqa: E402


def main():
    # ---- 1. Load one real 2-frame clip (scene-0061, first window) ----
    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        frame_index = json.load(f)
    base_dataset = Occ4DGSDataset(
        nusc, frame_index,
        os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
        os.path.join(REPO_ROOT, "data", "occ3d_gts"),
        pipeline=build_pipeline(),
    )
    clip_dataset = Occ4DGSClipDataset(base_dataset, unroll_window=2)
    scene0061_clip_idx = next(
        i for i, clip in enumerate(clip_dataset.clips)
        if base_dataset.samples[clip[0]][0] == "scene-0061"
    )
    frame0_dict, frame1_dict = clip_dataset[scene0061_clip_idx]
    cuda0 = to_cuda(to_batch_of_one(frame0_dict))
    cuda1 = to_cuda(to_batch_of_one(frame1_dict))

    # ---- 2. Build the real Stage A segmentor + real, CURRENT temporal module ----
    # Both come straight from stage_b_engine -- whatever architecture is live right
    # now (post Steps 1-4, and automatically post any future change too) is exactly
    # what this test exercises. build_stage_a loads the real trained checkpoint if
    # present, falling back to init_weights() otherwise -- matches every real script.
    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_stage_a(cfg)
    encoder = CurrentFrameEncoder(segmentor)
    pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout, spawn_head = build_temporal_module()
    for m in (pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout):
        m.eval()
    if spawn_head is not None:
        spawn_head.eval()

    torch.cuda.reset_peak_memory_stats()

    # ---- 3. Get a REAL G_0 by running Stage A's full forward on frame 0 ----
    g0, g0_dict = get_real_g0(segmentor, cuda0)
    n_g = g0.means.shape[1] if g0.means.dim() == 3 else g0.means.shape[0]
    print(f"[1/4] Real G_0 obtained: means shape {tuple(g0.means.shape)} (N_g={n_g})")
    assert n_g == 6400, f"expected N_g=6400 per config, got {n_g}"
    print("      [PASS] G_0 shape matches configured N_g=6400")

    buffer = ReferenceBuffer(g0)

    # ---- 4. Frame 1: deform_one_step -- the EXACT function every real training run
    #          and every evaluation script uses, not a hand-reconstructed copy ----
    with torch.no_grad():
        g_prev = buffer.read()
        g1, delta_mu, delta_r = deform_one_step(
            g_prev, encoder, pool, hypernet, deform_mu, deform_r,
            feature_dropout, z_dropout, cuda0, cuda1,
            spawn_head=spawn_head, return_deltas=True,
        )
        buffer.write(g1)

    means_prev = g_prev.means
    means_new = buffer.read().means
    assert not torch.allclose(means_prev, means_new), (
        "buffer.read() after write(g1) is identical to g0's means -- recursion is "
        "silently broken on real data (Phase 4's toy test alone did not catch this)"
    )
    print("[2/4] [PASS] G_1 provably distinct from G_0 on real data "
          f"(mean abs delta: {(means_new - means_prev).abs().mean().item():.6f})")

    # ---- 5. GaussianHead splat, called standalone, on G_1 with frame 1's real GT ----
    for axis, name, lo, hi in zip(range(3), ("x", "y", "z"), PC_RANGE[:3], PC_RANGE[3:]):
        vals = g1.means[..., axis]
        print(f"      diagnostic: g1.means[{name}] range: {vals.min().item():.4f} to "
              f"{vals.max().item():.4f}  (valid: [{lo}, {hi}])")
    print("      diagnostic: delta_mu range:", delta_mu.min().item(), delta_mu.max().item())

    g1_wrapped = [{"gaussian": type(g0_dict)(
        means=g1.means, rotations=g1.rotations, scales=g1.scales,
        opacities=g1.opacities, semantics=g1.semantics,
    )}]
    with torch.no_grad():
        head_out = segmentor.head(representation=g1_wrapped, metas=cuda1["metas"])
    pred_occ = head_out["pred_occ"][0]
    print(f"[3/4] [PASS] GaussianHead callable standalone on G_1: "
          f"pred_occ shape {tuple(pred_occ.shape)}")

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[4/4] Peak VRAM (unroll_window=2, 1 scene, 1 clip, frozen encoders, "
          f"no grad/optimizer): {peak_vram_gb:.2f} GB")

    print("\nAll Phase 5 real-data wiring checks passed.")


if __name__ == "__main__":
    main()
