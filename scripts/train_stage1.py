"""
scripts/train_stage1.py

Phase 5 step 3 (of 3, per this session's plan): the real stage_1_warmup training loop,
scaled up from train_stage1_smoketest.py's single-repeated-clip sanity check to real
per-epoch training across all clips in the given scene(s), plus the do-nothing-baseline
comparison the roadmap's Phase 5 exit checklist requires as first evidence the temporal
module is learning anything (not just capable of overfitting one clip).

Deliberate simplifications vs the full configs/stage_b_temporal.yaml schedule, each
logged so it's a visible decision, not silent drift:
  - Starts with N_EPOCHS=10 and SCENES=["scene-0061"] (roadmap step 4: "1-2 scenes only;
    profile VRAM before scaling to all 10" -- this run is that first step, not the final
    60-epoch/10-scene run).
  - LR schedule: torch.optim.lr_scheduler.CosineAnnealingLR, not timm's
    CosineLRScheduler (used in GaussianFormer3D's own train.py) -- functionally
    equivalent for this run's purposes (cosine decay to ~0), simpler dependency. Revisit
    if timm's warmup/restart behavior specifically matters later.
  - No AMP/fp16 yet (configs/stage_b_temporal.yaml specifies precision: amp_fp16) --
    added once a full-precision baseline run is confirmed working, so any future
    numerical issue can be isolated to AMP specifically rather than conflated with
    getting the training loop itself right.
  - Gradient accumulation matches the yaml (grad_accumulation_steps: 4).

Run from repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/train_stage1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_cwd = os.getcwd()
from run_stage_a_frame0 import build_pipeline, to_batch_of_one, GF3D_ROOT  # noqa: E402
os.chdir(_cwd)

import torch  # noqa: E402
from mmengine import Config  # noqa: E402
from mmseg.models import build_segmentor  # noqa: E402

sys.path.insert(0, GF3D_ROOT)
import model  # noqa: E402,F401
from loss import OPENOCC_LOSS  # noqa: E402
from misc.metric_util import MeanIoU  # noqa: E402

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
import wandb  # noqa: E402

# metric_util.MeanIoU calls wandb.log(...) internally and will raise if no run is
# active. mode="disabled" makes every wandb call in this process a local no-op --
# no login, no API key, no network call, nothing written to any account. This does
# NOT use or touch your labmate's wandb session (README's shared-machine note).
wandb.init(mode="disabled")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
CLASS_NAMES = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation',
]  # confirmed against train.py's MeanIoU construction (16 classes, 1..16)

# ---- Step 4 scope: 1-2 scenes first, per roadmap. Not yet the full 10. ----
SCENES = ["scene-0061"]
N_EPOCHS = 10
GRAD_ACCUM_STEPS = 4  # matches configs/stage_b_temporal.yaml stage_1_warmup
LR = 1e-4
WEIGHT_DECAY = 0.01


def to_cuda(batch):
    out = {"imgs": batch["imgs"].cuda(), "points": [t.cuda() for t in batch["points"]]}
    out["metas"] = {k: v.cuda() for k, v in batch["metas"].items()}
    out["dpt"] = batch["dpt"].cuda() if batch["dpt"] is not None else None
    return out


def build_stage_a(cfg):
    segmentor = build_segmentor(cfg.model)
    checkpoint_path = os.path.join(
        REPO_ROOT, "experiments", "stage_a_checkpoints", "stage_a_best.pth"
    )
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        segmentor.load_state_dict(state_dict, strict=True)
        print(f"Loaded real trained Stage A checkpoint: {checkpoint_path}")
    else:
        segmentor.init_weights()
        print("WARNING: no trained Stage A checkpoint found, falling back to "
              "init_weights() -- this is the untrained-G_0 state that caused the "
              "original exit-checklist failure. Run scripts/train_stage_a.py first.")
    segmentor = segmentor.cuda()
    segmentor.eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)
    return segmentor


def build_temporal_module():
    pool = PoolFeatures(img_channels=128, dpt_channels=112, num_levels=4, in_dim=128).cuda()
    hypernet = MotionHyperNet(in_dim=128, grid_feat_dim=16, resolutions=(4, 8, 16)).cuda()
    deform_mu = DeformHeadMu(in_dim=3 * (16 + 6), hidden_dim=128).cuda()
    deform_r = DeformHeadR(in_dim=3 * (16 + 6), hidden_dim=128, max_angle_rad=0.3).cuda()
    return pool, hypernet, deform_mu, deform_r


def get_real_g0(segmentor, cuda0):
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
    return g0, g0_dict


def deform_one_step(g_prev, encoder, pool, hypernet, deform_mu, deform_r, cuda_frame,
                     no_grad_encoder=True):
    """Runs frame t's encoder (frozen, always no_grad) + the trainable temporal module
    (grad-tracked unless the caller is itself inside a no_grad context, e.g. eval)."""
    ctx = torch.no_grad() if no_grad_encoder else torch.enable_grad()
    with ctx:
        ms_img_feats, _dpt_dist, out_dpt_multiscale = encoder.encode(
            cuda_frame["imgs"], cuda_frame["dpt"], cuda_frame["metas"]
        )
    pooled = pool(ms_img_feats, out_dpt_multiscale)
    grids = hypernet(pooled)
    means_flat = g_prev.means.squeeze(0) if g_prev.means.dim() == 3 else g_prev.means
    z = query_motion_grid(means_flat, grids, PC_RANGE)
    delta_mu = deform_mu(z)
    delta_r = deform_r(z)
    return apply_update_rule(g_prev, delta_mu, delta_r, pc_range=PC_RANGE)


def splat_and_loss(segmentor, g_state, g_dict_type, cuda_frame, cfg, loss_func):
    g_wrapped = [{"gaussian": g_dict_type(
        means=g_state.means, rotations=g_state.rotations, scales=g_state.scales,
        opacities=g_state.opacities, semantics=g_state.semantics,
    )}]
    head_out = segmentor.head(representation=g_wrapped, metas=cuda_frame["metas"])
    loss_input = {}
    for k, v in cfg.loss_input_convertion.items():
        loss_input.update({k: head_out[v]})
    loss, loss_dict = loss_func(loss_input)
    return loss, loss_dict, head_out


def main():
    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        full_frame_index = json.load(f)
    frame_index = {s: full_frame_index[s] for s in SCENES}

    base_dataset = Occ4DGSDataset(
        nusc, frame_index,
        os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
        os.path.join(REPO_ROOT, "data", "occ3d_gts"),
        pipeline=build_pipeline(),
    )
    clip_dataset = Occ4DGSClipDataset(base_dataset, unroll_window=2)
    print(f"{SCENES}: {len(clip_dataset)} clips at unroll_window=2, {N_EPOCHS} epochs")

    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_stage_a(cfg)
    encoder = CurrentFrameEncoder(segmentor)
    pool, hypernet, deform_mu, deform_r = build_temporal_module()

    trainable_params = (
        list(pool.parameters()) + list(hypernet.parameters())
        + list(deform_mu.parameters()) + list(deform_r.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = N_EPOCHS * len(clip_dataset)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    torch.cuda.reset_peak_memory_stats()
    step = 0
    for epoch in range(N_EPOCHS):
        epoch_losses = []
        optimizer.zero_grad()
        for clip_idx in range(len(clip_dataset)):
            frame0_dict, frame1_dict = clip_dataset[clip_idx]
            cuda0 = to_cuda(to_batch_of_one(frame0_dict))
            cuda1 = to_cuda(to_batch_of_one(frame1_dict))

            g0, g0_dict = get_real_g0(segmentor, cuda0)
            buffer = ReferenceBuffer(g0)
            g1 = deform_one_step(buffer.read(), encoder, pool, hypernet,
                                  deform_mu, deform_r, cuda1)
            buffer.write(g1)

            loss, loss_dict, _ = splat_and_loss(
                segmentor, buffer.read(), type(g0_dict), cuda1, cfg, loss_func
            )
            (loss / GRAD_ACCUM_STEPS).backward()
            step += 1
            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=35)  # matches
                                                                                 # Stage A's own grad_max_norm
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            epoch_losses.append(loss.item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"epoch {epoch:3d}  mean_loss={mean_loss:.5f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  peak_vram={peak_vram_gb:.2f}GB")

    # Save the trained temporal module (pool/hypernet/deform_mu/deform_r) so a
    # separate held-out-scene evaluation can load these exact weights without
    # retraining -- retraining on held-out data would just be a second in-sample fit,
    # not a generalization test.
    temporal_ckpt_dir = os.path.join(REPO_ROOT, "experiments", "stage_b_temporal_checkpoints")
    os.makedirs(temporal_ckpt_dir, exist_ok=True)
    temporal_ckpt_path = os.path.join(temporal_ckpt_dir, "stage1_warmup_temporal.pth")
    torch.save({
        "pool": pool.state_dict(),
        "hypernet": hypernet.state_dict(),
        "deform_mu": deform_mu.state_dict(),
        "deform_r": deform_r.state_dict(),
        "trained_on_scenes": SCENES,
        "n_epochs": N_EPOCHS,
    }, temporal_ckpt_path)
    print(f"Saved trained temporal module to {temporal_ckpt_path}")

    # ---- Do-nothing baseline comparison (exit checklist item 2) ----
    print("\n--- Do-nothing baseline (Delta_mu=0, Delta_r=identity) vs trained, "
          f"per-frame mIoU across all {len(clip_dataset)} clips in {SCENES} ---")

    miou_trained = MeanIoU(list(range(1, 17)), 17, CLASS_NAMES, True, 17, filter_minmax=False)
    miou_donothing = MeanIoU(list(range(1, 17)), 17, CLASS_NAMES, True, 17, filter_minmax=False)
    miou_trained.reset()
    miou_donothing.reset()

    with torch.no_grad():
        for clip_idx in range(len(clip_dataset)):
            frame0_dict, frame1_dict = clip_dataset[clip_idx]
            cuda0 = to_cuda(to_batch_of_one(frame0_dict))
            cuda1 = to_cuda(to_batch_of_one(frame1_dict))

            g0, g0_dict = get_real_g0(segmentor, cuda0)

            # trained
            buffer = ReferenceBuffer(g0)
            g1_trained = deform_one_step(buffer.read(), encoder, pool, hypernet,
                                          deform_mu, deform_r, cuda1)
            _, _, head_out_trained = splat_and_loss(
                segmentor, g1_trained, type(g0_dict), cuda1, cfg, loss_func
            )

            # do-nothing: G_1 == G_0 verbatim, no deform at all
            _, _, head_out_donothing = splat_and_loss(
                segmentor, g0, type(g0_dict), cuda1, cfg, loss_func
            )

            gt_occ = head_out_trained["sampled_label"][0]  # (N,) -- batch idx 0
            mask = head_out_trained["occ_mask"].flatten(1)[0].bool()

            pred_trained = head_out_trained["pred_occ"][-1][0].argmax(0)
            pred_donothing = head_out_donothing["pred_occ"][-1][0].argmax(0)

            miou_trained._after_step(pred_trained, gt_occ, mask)
            miou_donothing._after_step(pred_donothing, gt_occ, mask)

    miou_t, iou2_t = miou_trained._after_epoch()
    miou_d, iou2_d = miou_donothing._after_epoch()
    print(f"Trained:    mIoU={miou_t}, iou2={iou2_t}")
    print(f"Do-nothing: mIoU={miou_d}, iou2={iou2_d}")
    print(f"\nExit checklist item 2 (trained meaningfully above do-nothing baseline): "
          f"{'NEEDS MORE TRAINING/INSPECTION' if not (miou_t and miou_d) else 'see numbers above'}")


if __name__ == "__main__":
    main()