"""
scripts/train_stage1.py

Phase 5 step 3: real stage_1_warmup training loop.
History of fixes, in order: held-out tracking (Step 0) -> delta-feature conditioning
(Step 2, fixed the absolute-scene-appearance shortcut) -> dropout/weight-decay/motion-
penalty regularization (Step 1, no effect) -> PE(mu) removal ablation (no effect).

THIS VERSION: fixes a confound discovered in the 1/3/6/8-scene sweep -- held-out
checks were gated by EPOCH index, but epoch != amount of training when clip counts
differ 8x across scene counts (n=1: 38 clips/epoch; n=8: 316 clips/epoch). By "epoch 1"
n=8 had done ~8x more optimizer steps than n=1, so the sweep's apparent "more scenes ->
worse held-out delta" trend was confounded with "more optimizer steps -> worse held-out
delta" (which every single prior experiment, regardless of what else changed, has shown
as a consistent decay curve). Fix: gate held-out checks by OPTIMIZER STEP COUNT, not
epoch, so all scene-count runs are compared on the same step axis.

Usage:
    PYTHONNOUSERSITE=1 python scripts/train_stage1.py [N_SCENES]
    -- N_SCENES (default: all 8) selects the first N scenes of ALL_TRAINED_SCENES
       (neutral order), all evaluated against the SAME already-trained 8-scene Stage A
       checkpoint and the SAME fixed seed, for the scene-scaling sweep.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_cwd = os.getcwd()
from run_stage_a_frame0 import build_pipeline, to_batch_of_one, GF3D_ROOT  # noqa: E402
os.chdir(_cwd)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
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
    compute_relative_transform,
    apply_ego_compensated_update_rule,
)
from src.models.stage_b_temporal.current_frame_encoder import CurrentFrameEncoder  # noqa: E402
from src.models.stage_b_temporal.pool_features import PoolFeatures  # noqa: E402

import json  # noqa: E402
import wandb  # noqa: E402

wandb.init(mode="disabled")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
CLASS_NAMES = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation',
]

SEED = 42

ALL_TRAINED_SCENES = ["scene-0061", "scene-0103", "scene-0553", "scene-0655",
                      "scene-0757", "scene-0796", "scene-0916", "scene-1077"]
_n_scenes_arg = int(sys.argv[1]) if len(sys.argv) > 1 else len(ALL_TRAINED_SCENES)
SCENES = ALL_TRAINED_SCENES[:_n_scenes_arg]

N_EPOCHS = 20
HELDOUT_SCENES = ["scene-1094", "scene-1100"]
EVAL_EVERY_OPT_STEPS = 20  # STEP-based, not epoch-based (see module docstring)
GRAD_ACCUM_STEPS = 4
LR = 1e-4
WEIGHT_DECAY = 0.05
DROPOUT_P = 0.2
MOTION_PENALTY_WEIGHT = 0.01


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
        print("WARNING: no trained Stage A checkpoint found, falling back to init_weights().")
    segmentor = segmentor.cuda()
    segmentor.eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)
    return segmentor


def build_temporal_module():
    pool = PoolFeatures(img_channels=128, dpt_channels=112, num_levels=4, in_dim=128).cuda()
    hypernet = MotionHyperNet(in_dim=128, grid_feat_dim=16, resolutions=(4, 8, 16)).cuda()
    deform_mu = DeformHeadMu(in_dim=3 * 16, hidden_dim=128).cuda()
    deform_r = DeformHeadR(in_dim=3 * 16, hidden_dim=128, max_angle_rad=0.3).cuda()
    feature_dropout = nn.Dropout(p=DROPOUT_P).cuda()
    z_dropout = nn.Dropout(p=DROPOUT_P).cuda()
    return pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout


def get_real_g0(segmentor, cuda0):
    with torch.no_grad():
        representation0 = segmentor(
            imgs=cuda0["imgs"].clone(),
            metas=cuda0["metas"],
            points=cuda0["points"],
            dpt=cuda0["dpt"].clone() if cuda0["dpt"] is not None else None,
            rep_only=True,
        )
    g0_dict = representation0[-1]["gaussian"]
    g0 = GaussianState(
        means=g0_dict.means, rotations=g0_dict.rotations,
        scales=g0_dict.scales, opacities=g0_dict.opacities, semantics=g0_dict.semantics,
    )
    return g0, g0_dict


def deform_one_step(g_prev, encoder, pool, hypernet, deform_mu, deform_r,
                     feature_dropout, z_dropout, cuda_prev, cuda_curr,
                     no_grad_encoder=True, return_deltas=False):
    ctx = torch.no_grad() if no_grad_encoder else torch.enable_grad()
    with ctx:
        ms_img_feats_prev, _dpt_dist_prev, out_dpt_multiscale_prev = encoder.encode(
            cuda_prev["imgs"], cuda_prev["dpt"], cuda_prev["metas"]
        )
        ms_img_feats_curr, _dpt_dist_curr, out_dpt_multiscale_curr = encoder.encode(
            cuda_curr["imgs"], cuda_curr["dpt"], cuda_curr["metas"]
        )
    pooled_prev = pool(ms_img_feats_prev, out_dpt_multiscale_prev)
    pooled_curr = pool(ms_img_feats_curr, out_dpt_multiscale_curr)
    pooled_delta = feature_dropout(pooled_curr - pooled_prev)

    grids = hypernet(pooled_delta)
    means_flat = g_prev.means.squeeze(0) if g_prev.means.dim() == 3 else g_prev.means
    z = z_dropout(query_motion_grid(means_flat, grids, PC_RANGE, include_pe=False))
    delta_mu = deform_mu(z)
    delta_r = deform_r(z)

    pose_prev = cuda_prev["metas"]["lidar2global"][0]
    pose_curr = cuda_curr["metas"]["lidar2global"][0]
    relative_transform = compute_relative_transform(pose_prev, pose_curr)
    g_t = apply_ego_compensated_update_rule(
        g_prev, delta_mu, delta_r, relative_transform, pc_range=PC_RANGE
    )
    if return_deltas:
        return g_t, delta_mu, delta_r
    return g_t


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


def adjusted_miou(miou_metric, class_names):
    total_seen = miou_metric.total_seen[:-1]
    total_correct = miou_metric.total_correct[:-1]
    total_positive = miou_metric.total_positive[:-1]
    ious = []
    for i in range(len(class_names)):
        if total_seen[i].item() == 0:
            continue
        union = (total_seen[i] + total_positive[i] - total_correct[i]).item()
        ious.append((total_correct[i].item() / union) if union > 0 else 0.0)
    return (sum(ious) / len(ious) * 100) if ious else float("nan")


def build_heldout_clip_datasets(nusc, full_frame_index):
    datasets = []
    for scene_name in HELDOUT_SCENES:
        frame_index = {scene_name: full_frame_index[scene_name]}
        base = Occ4DGSDataset(
            nusc, frame_index,
            os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
            os.path.join(REPO_ROOT, "data", "occ3d_gts"),
            pipeline=build_pipeline(),
        )
        datasets.append(Occ4DGSClipDataset(base, unroll_window=2))
    return datasets


def evaluate_heldout(segmentor, encoder, pool, hypernet, deform_mu, deform_r,
                      feature_dropout, z_dropout, cfg, loss_func,
                      heldout_clip_datasets, mode):
    assert mode in ("trained", "donothing", "ego_only")
    modules = (pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout)
    for m in modules:
        m.eval()
    miou = MeanIoU(list(range(1, 17)), 17, CLASS_NAMES, True, 17, filter_minmax=False)
    miou.reset()
    with torch.no_grad():
        for clip_dataset in heldout_clip_datasets:
            for clip_idx in range(len(clip_dataset)):
                frame0_dict, frame1_dict = clip_dataset[clip_idx]
                cuda0 = to_cuda(to_batch_of_one(frame0_dict))
                cuda1 = to_cuda(to_batch_of_one(frame1_dict))
                g0, g0_dict = get_real_g0(segmentor, cuda0)
                if mode == "trained":
                    g1 = deform_one_step(g0, encoder, pool, hypernet, deform_mu, deform_r,
                                          feature_dropout, z_dropout, cuda0, cuda1)
                elif mode == "ego_only":
                    pose_prev = cuda0["metas"]["lidar2global"][0]
                    pose_curr = cuda1["metas"]["lidar2global"][0]
                    relative_transform = compute_relative_transform(pose_prev, pose_curr)
                    zero_mu = torch.zeros(g0.means.shape[-2], 3, device=g0.means.device)
                    zero_r = torch.zeros(g0.means.shape[-2], 4, device=g0.means.device)
                    zero_r[:, 0] = 1.0
                    g1 = apply_ego_compensated_update_rule(
                        g0, zero_mu, zero_r, relative_transform, pc_range=PC_RANGE
                    )
                else:
                    g1 = g0
                _, _, head_out = splat_and_loss(segmentor, g1, type(g0_dict), cuda1, cfg, loss_func)
                gt_occ = head_out["sampled_label"][0]
                mask = head_out["occ_mask"].flatten(1)[0].bool()
                pred = head_out["pred_occ"][-1][0].argmax(0)
                miou._after_step(pred, gt_occ, mask)
    for m in modules:
        m.train()
    return adjusted_miou(miou, CLASS_NAMES)


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
    print(f"{SCENES}: {len(clip_dataset)} clips at unroll_window=2, {N_EPOCHS} epochs, "
          f"eval every {EVAL_EVERY_OPT_STEPS} OPTIMIZER STEPS (not epochs)")

    heldout_clip_datasets = build_heldout_clip_datasets(nusc, full_frame_index)
    print(f"Held-out (never trained): {HELDOUT_SCENES}, "
          f"{sum(len(d) for d in heldout_clip_datasets)} clips total")

    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_stage_a(cfg)
    encoder = CurrentFrameEncoder(segmentor)
    torch.manual_seed(SEED)
    pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout = build_temporal_module()
    loss_func_for_baseline = OPENOCC_LOSS.build(cfg.loss).cuda()

    heldout_donothing = evaluate_heldout(
        segmentor, encoder, pool, hypernet, deform_mu, deform_r,
        feature_dropout, z_dropout, cfg, loss_func_for_baseline,
        heldout_clip_datasets, mode="donothing"
    )
    print(f"Held-out do-nothing baseline (adjusted mIoU, fixed reference): "
          f"{heldout_donothing:.3f}")

    trainable_params = (
        list(pool.parameters()) + list(hypernet.parameters())
        + list(deform_mu.parameters()) + list(deform_r.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = N_EPOCHS * len(clip_dataset)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    best_heldout_delta = float("-inf")
    best_state = None
    opt_step_count = 0
    next_eval_at = EVAL_EVERY_OPT_STEPS

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
            g1, delta_mu, delta_r = deform_one_step(
                buffer.read(), encoder, pool, hypernet, deform_mu, deform_r,
                feature_dropout, z_dropout, cuda0, cuda1, return_deltas=True
            )
            buffer.write(g1)

            task_loss, loss_dict, _ = splat_and_loss(
                segmentor, buffer.read(), type(g0_dict), cuda1, cfg, loss_func
            )
            motion_penalty = delta_mu.pow(2).mean() + (1.0 - delta_r[..., 0].abs()).mean()
            loss = task_loss + MOTION_PENALTY_WEIGHT * motion_penalty

            (loss / GRAD_ACCUM_STEPS).backward()
            step += 1
            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=35)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                opt_step_count += 1

                if opt_step_count >= next_eval_at:
                    heldout_trained = evaluate_heldout(
                        segmentor, encoder, pool, hypernet, deform_mu, deform_r,
                        feature_dropout, z_dropout, cfg, loss_func,
                        heldout_clip_datasets, mode="trained"
                    )
                    delta = heldout_trained - heldout_donothing
                    line = (f"[opt_step {opt_step_count:5d}] (epoch {epoch:3d})  "
                            f"held-out adj.mIoU: trained={heldout_trained:.3f} "
                            f"vs do-nothing={heldout_donothing:.3f} (delta={delta:+.3f})")
                    if delta > best_heldout_delta:
                        best_heldout_delta = delta
                        best_state = {
                            "pool": {k: v.clone() for k, v in pool.state_dict().items()},
                            "hypernet": {k: v.clone() for k, v in hypernet.state_dict().items()},
                            "deform_mu": {k: v.clone() for k, v in deform_mu.state_dict().items()},
                            "deform_r": {k: v.clone() for k, v in deform_r.state_dict().items()},
                            "opt_step": opt_step_count,
                            "heldout_delta": delta,
                        }
                        line += "  -> new best held-out delta, checkpointed"
                    print(line)
                    next_eval_at += EVAL_EVERY_OPT_STEPS

            epoch_losses.append(task_loss.item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"epoch {epoch:3d}  mean_loss={mean_loss:.5f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  peak_vram={peak_vram_gb:.2f}GB  "
              f"(opt_steps so far: {opt_step_count})")

    temporal_ckpt_dir = os.path.join(REPO_ROOT, "experiments", "stage_b_temporal_checkpoints")
    os.makedirs(temporal_ckpt_dir, exist_ok=True)

    final_ckpt_path = os.path.join(
        temporal_ckpt_dir, f"stage1_warmup_temporal_n{len(SCENES)}_final.pth")
    torch.save({
        "pool": pool.state_dict(), "hypernet": hypernet.state_dict(),
        "deform_mu": deform_mu.state_dict(), "deform_r": deform_r.state_dict(),
        "trained_on_scenes": SCENES, "n_epochs": N_EPOCHS,
        "total_opt_steps": opt_step_count,
    }, final_ckpt_path)
    print(f"Saved final-step temporal module to {final_ckpt_path}")

    if best_state is not None:
        best_ckpt_path = os.path.join(
            temporal_ckpt_dir, f"stage1_warmup_temporal_n{len(SCENES)}.pth")
        torch.save({
            "pool": best_state["pool"], "hypernet": best_state["hypernet"],
            "deform_mu": best_state["deform_mu"], "deform_r": best_state["deform_r"],
            "trained_on_scenes": SCENES, "n_epochs": N_EPOCHS,
            "best_opt_step": best_state["opt_step"], "best_heldout_delta": best_state["heldout_delta"],
        }, best_ckpt_path)
        print(f"Saved BEST-held-out-delta temporal module "
              f"(opt_step {best_state['opt_step']}, delta={best_state['heldout_delta']:+.3f}) "
              f"to {best_ckpt_path}")

    print(f"\nSUMMARY for SCENES={SCENES} ({len(SCENES)} scenes, {len(clip_dataset)} "
          f"clips/epoch, {opt_step_count} total optimizer steps): "
          f"best held-out delta = {best_heldout_delta:+.3f} at opt_step "
          f"{best_state['opt_step'] if best_state else 'N/A'}")


if __name__ == "__main__":
    main()
