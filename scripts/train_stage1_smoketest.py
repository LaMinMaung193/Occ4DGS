"""
scripts/train_stage1_smoketest.py

Phase 5 step 1 (of 3, per this session's plan): minimal training-loop smoke test before
writing the full stage_1_warmup schedule. No scheduler, no checkpointing, no wandb, one
scene only. Confirms:
  1. loss.backward() runs without error through the real Stage A -> Stage B -> GaussianHead
     -> OccupancyLoss chain.
  2. Gradients reach PoolFeatures / MotionHyperNet / DeformHeadMu / DeformHeadR (the
     temporal module) and do NOT reach the frozen Stage A submodules
     (img_backbone/img_neck/pts_dpt_head/lifter/encoder/head's own parameters).
  3. Loss visibly decreases over a handful of steps on one repeated clip (the simplest
     possible "is this actually learning anything" signal, before the do-nothing-baseline
     comparison or the full 60-epoch schedule).

Reuses OPENOCC_LOSS / cfg.loss / cfg.loss_input_convertion exactly as GaussianFormer3D's
own train.py does (confirmed via source read, EXPERIMENT_LOG.md Phase 5 bridge
investigation) -- NOT a reimplemented loss call.

Run from repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/train_stage1_smoketest.py
(PYTHONNOUSERSITE=1 is required for `from loss import OPENOCC_LOSS` to work, per the
known environment quirk documented in README.md.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_cwd = os.getcwd()
from src.datasets.gf3d_pipeline import build_pipeline, to_batch_of_one, GF3D_ROOT  # noqa: E402
os.chdir(_cwd)  # undo run_stage_a_frame0's module-level os.chdir(GF3D_ROOT) side effect

import torch  # noqa: E402
from mmengine import Config  # noqa: E402
from mmseg.models import build_segmentor  # noqa: E402

sys.path.insert(0, GF3D_ROOT)
import model  # noqa: E402,F401  -- triggers @MODELS.register_module() decorators
from loss import OPENOCC_LOSS  # noqa: E402  -- requires PYTHONNOUSERSITE=1, see docstring

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
N_STEPS = 30  # smoke test only -- full stage_1_warmup is 60 epochs, written separately


def main():
    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        frame_index = json.load(f)

    # Step 1 scope: ONE scene only, per this session's plan (full stage_1_warmup scales
    # to 1-2 scenes then all 10, per roadmap Phase 5 step 4 -- not attempted here yet).
    frame_index = {"scene-0061": frame_index["scene-0061"]}

    base_dataset = Occ4DGSDataset(
        nusc, frame_index,
        os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
        os.path.join(REPO_ROOT, "data", "occ3d_gts"),
        pipeline=build_pipeline(),
    )
    clip_dataset = Occ4DGSClipDataset(base_dataset, unroll_window=2)
    print(f"scene-0061: {len(clip_dataset)} clips available at unroll_window=2")

    # ---- Build Stage A, freeze everything (per stage_1_warmup: "Stage A frozen") ----
    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_segmentor(cfg.model)
    segmentor.init_weights()
    segmentor = segmentor.cuda()
    segmentor.eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)

    # ---- Build the temporal module (trainable) ----
    encoder = CurrentFrameEncoder(segmentor)
    pool = PoolFeatures(img_channels=128, dpt_channels=112, num_levels=4, in_dim=128).cuda()
    hypernet = MotionHyperNet(in_dim=128, grid_feat_dim=16, resolutions=(4, 8, 16)).cuda()
    deform_mu = DeformHeadMu(in_dim=3 * (16 + 6), hidden_dim=128).cuda()
    deform_r = DeformHeadR(in_dim=3 * (16 + 6), hidden_dim=128, max_angle_rad=0.3).cuda()

    trainable_params = (
        list(pool.parameters()) + list(hypernet.parameters())
        + list(deform_mu.parameters()) + list(deform_r.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    # ---- Loss, built exactly as GaussianFormer3D's own train.py does ----
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    # ---- One repeated clip for the smoke test (simplest possible learning signal) ----
    frame0_dict, frame1_dict = clip_dataset[0]
    batch0 = to_batch_of_one(frame0_dict)
    batch1 = to_batch_of_one(frame1_dict)

    def to_cuda(batch):
        out = {"imgs": batch["imgs"].cuda(), "points": [t.cuda() for t in batch["points"]]}
        out["metas"] = {k: v.cuda() for k, v in batch["metas"].items()}
        out["dpt"] = batch["dpt"].cuda() if batch["dpt"] is not None else None
        return out

    cuda0 = to_cuda(batch0)
    cuda1 = to_cuda(batch1)

    # Real G_0: frozen Stage A, no grad needed at all (never backprop through this).
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

    print(f"\nRunning {N_STEPS} steps on one repeated clip (scene-0061, clip 0)...")
    losses = []
    for step in range(N_STEPS):
        buffer = ReferenceBuffer(g0)  # fresh buffer each step -- smoke test only,
                                        # NOT truncated-BPTT across steps (that's the
                                        # full training loop's job, not this one's)

        with torch.no_grad():  # frozen encoder -- no grad needed through this either
            ms_img_feats, _dpt_dist, out_dpt_multiscale = encoder.encode(
                cuda1["imgs"], cuda1["dpt"], cuda1["metas"]
            )

        pooled = pool(ms_img_feats, out_dpt_multiscale)
        grids = hypernet(pooled)

        g_prev = buffer.read()
        means_flat = g_prev.means.squeeze(0) if g_prev.means.dim() == 3 else g_prev.means
        z = query_motion_grid(means_flat, grids, PC_RANGE)
        delta_mu = deform_mu(z)
        delta_r = deform_r(z)
        g1 = apply_update_rule(g_prev, delta_mu, delta_r, pc_range=PC_RANGE)
        buffer.write(g1)

        g1_wrapped = [{"gaussian": type(g0_dict)(
            means=g1.means, rotations=g1.rotations, scales=g1.scales,
            opacities=g1.opacities, semantics=g1.semantics,
        )}]
        head_out = segmentor.head(representation=g1_wrapped, metas=cuda1["metas"])

        loss_input = {}
        for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
            loss_input.update({loss_input_key: head_out[loss_input_val]})
        loss, loss_dict = loss_func(loss_input)

        optimizer.zero_grad()
        loss.backward()

        # ---- Gradient isolation check: frozen Stage A must get NO gradient ----
        if step == 0:
            frozen_grad_found = any(
                p.grad is not None and p.grad.abs().sum().item() > 0
                for p in segmentor.parameters()
            )
            assert not frozen_grad_found, (
                "A frozen Stage A parameter received a nonzero gradient -- something is "
                "differentiating through the wrong path. Do not proceed to the full "
                "training schedule until this is fixed."
            )
            trainable_has_grad = all(
                p.grad is not None for p in trainable_params
            )
            assert trainable_has_grad, (
                "At least one temporal-module parameter received NO gradient -- check "
                "PoolFeatures/MotionHyperNet/DeformHeadMu/DeformHeadR are actually wired "
                "into the loss, not accidentally detached somewhere."
            )
            print("  [PASS] step 0 gradient isolation: frozen Stage A untouched, "
                  "all temporal-module params received gradients")

        optimizer.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == N_STEPS - 1:
            detail = ", ".join(f"{k}: {v:.5f}" for k, v in loss_dict.items())
            print(f"  step {step:3d}  loss={loss.item():.5f}  ({detail})")

    print(f"\nLoss: first={losses[0]:.5f}  last={losses[-1]:.5f}  "
          f"delta={losses[0] - losses[-1]:+.5f}")
    if losses[-1] < losses[0]:
        print("Smoke test: loss decreased over the run -- wiring trains, as expected "
              "for a repeated single clip. (Not yet evidence of generalization -- that's "
              "the do-nothing-baseline comparison + full schedule, done separately.)")
    else:
        print("WARNING: loss did NOT decrease over the run. Before scaling to the full "
              "schedule, check LR, gradient isolation above, and whether N_STEPS=30 is "
              "simply too few for this loss/data -- do not assume the wiring is broken "
              "without checking these first.")


if __name__ == "__main__":
    main()