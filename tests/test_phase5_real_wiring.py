"""
tests/test_phase5_real_wiring.py

Phase 5 wiring validation: confirms the full chain -- real Stage A G_0, real frame-1
encoder features, PoolFeatures -> MotionHyperNet -> grid_query -> DeformHeadMu/R ->
apply_update_rule -> ReferenceBuffer -> GaussianHead splat -- actually runs end to end
on one real 2-frame clip, before any training loop is written.

This is NOT the Stage 1 warmup training script (that comes after this passes, per
roadmap Phase 5 step 3) -- no optimizer step happens here. This test only confirms:
  1. Real G_0 (from an actual Stage A forward pass) has the expected shape/fields.
  2. Frame 1's deformed G_1 is provably distinct from G_0 (buffer recursion holds on
     real data, not just the Phase 4 toy sequence).
  3. GaussianHead is callable standalone on G_1 with frame 1's own GT metas, producing
     pred_occ of the expected shape (confirms EXPERIMENT_LOG.md's Phase 5 bridge finding
     #3 -- standalone splat callability -- against real tensors, not just source reading).
  4. Peak VRAM at unroll_window=2, one scene, one clip -- roadmap Phase 5 step 4's exit
     checklist requirement to profile before scaling to 10 scenes / window=3.

Must be run from the repo root, in the gf3d conda env, e.g.:
    python tests/test_phase5_real_wiring.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

_cwd = os.getcwd()
from run_stage_a_frame0 import build_pipeline, to_batch_of_one, GF3D_ROOT  # noqa: E402
os.chdir(_cwd)  # undo run_stage_a_frame0's module-level os.chdir(GF3D_ROOT) side effect

import torch  # noqa: E402
from mmengine import Config  # noqa: E402
from mmseg.models import build_segmentor  # noqa: E402

sys.path.insert(0, GF3D_ROOT)
import model  # noqa: E402,F401  -- triggers @MODELS.register_module() decorators

from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402
from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402
from src.datasets.occ4dgs_clip_dataset import Occ4DGSClipDataset  # noqa: E402
from src.models.stage_b_temporal import (  # noqa: E402
    GaussianState,
    ReferenceBuffer,
    MotionHyperNet,
    query_motion_grid,
    DeformHeadMu,
    DeformHeadR,
    apply_update_rule,
)
from src.models.stage_b_temporal.current_frame_encoder import CurrentFrameEncoder  # noqa: E402
from src.models.stage_b_temporal.pool_features import PoolFeatures  # noqa: E402

import json  # noqa: E402


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]  # Occ3D range, confirmed Phase 0


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
    batch0 = to_batch_of_one(frame0_dict)
    batch1 = to_batch_of_one(frame1_dict)

    # ---- 2. Build the full segmentor, load pretrained weights, freeze encoders ----
    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_segmentor(cfg.model)
    segmentor.init_weights()
    segmentor = segmentor.cuda().eval()
    for submodule in (segmentor.img_backbone, segmentor.img_neck, segmentor.pts_dpt_head):
        for p in submodule.parameters():
            p.requires_grad_(False)

    def to_cuda(batch):
        out = {"imgs": batch["imgs"].cuda(), "points": [t.cuda() for t in batch["points"]]}
        out["metas"] = {k: v.cuda() for k, v in batch["metas"].items()}
        out["dpt"] = batch["dpt"].cuda() if batch["dpt"] is not None else None
        return out

    cuda0 = to_cuda(batch0)
    cuda1 = to_cuda(batch1)

    torch.cuda.reset_peak_memory_stats()

    # ---- 3. Get a REAL G_0 by running Stage A's full forward on frame 0 ----
    with torch.no_grad():
        representation0 = segmentor(
            imgs=cuda0["imgs"], metas=cuda0["metas"], points=cuda0["points"],
            dpt=cuda0["dpt"], rep_only=True,
        )
    g0_dict = representation0[-1]["gaussian"]
    g0 = GaussianState(
        means=g0_dict.means, rotations=g0_dict.rotations,
        scales=g0_dict.scales, opacities=g0_dict.opacities, semantics=g0_dict.semantics,
    )
    n_g = g0.means.shape[1] if g0.means.dim() == 3 else g0.means.shape[0]
    print(f"[1/4] Real G_0 obtained: means shape {tuple(g0.means.shape)} (N_g={n_g})")
    assert n_g == 6400, f"expected N_g=6400 per config, got {n_g}"
    print("      [PASS] G_0 shape matches configured N_g=6400")

    buffer = ReferenceBuffer(g0)

    # ---- 4. Frame 1: real encoder features -> pool -> hypernet -> deform -> G_1 ----
    encoder = CurrentFrameEncoder(segmentor)
    with torch.no_grad():
        ms_img_feats, _dpt_dist, out_dpt_multiscale = encoder.encode(
            cuda1["imgs"], cuda1["dpt"], cuda1["metas"]
        )

    img_channels = ms_img_feats[0].shape[2]
    dpt_channels = out_dpt_multiscale[0].shape[2]
    pool = PoolFeatures(img_channels=img_channels, dpt_channels=dpt_channels,
                         num_levels=len(ms_img_feats), in_dim=128).cuda()
    hypernet = MotionHyperNet(in_dim=128, grid_feat_dim=16, resolutions=(4, 8, 16)).cuda()
    deform_mu = DeformHeadMu(in_dim=3 * (16 + 6), hidden_dim=128).cuda()
    deform_r = DeformHeadR(in_dim=3 * (16 + 6), hidden_dim=128, max_angle_rad=0.3).cuda()

    with torch.no_grad():
        pooled = pool(ms_img_feats, out_dpt_multiscale)
        grids = hypernet(pooled)

        g_prev = buffer.read()
        means_flat = g_prev.means.squeeze(0) if g_prev.means.dim() == 3 else g_prev.means
        z = query_motion_grid(means_flat, grids, PC_RANGE)
        delta_mu = deform_mu(z)
        delta_r = deform_r(z)
        g1 = apply_update_rule(g_prev, delta_mu, delta_r, pc_range=PC_RANGE)
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