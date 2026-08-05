"""
scripts/train_stage1.py

Phase 5 step 3: real stage_1_warmup training loop.
History (see EXPERIMENT_LOG.md for full detail): held-out tracking (Step 0) ->
delta-feature conditioning (Step 2) -> regularization (Step 1, no effect) -> PE
removal (no effect) -> step-matched scene-scaling sweep (found the fixed-Gaussian-
budget representational gap, confirmed the majority regime at 44.4% of clips) ->
ego-motion compensation (mechanism confirmed correct via near-zero-motion-scene test,
but combined with the learned residual it made held-out results WORSE than
do-nothing, because opacity-zeroing throws away real information the residual cannot
recover) -> heuristic recycling (no improvement) -> learned SpawnHead -> Steps 1-4
(rotation-order fix, grid rebalance, PE-as-coordinate query, ConvHyperNet).

Phase 5 repo-layout cleanup: all reusable engine code (build_stage_a,
build_temporal_module, deform_one_step, evaluate_heldout, etc. -- everything 4 other
scripts depend on) moved to src/training/stage_b_engine.py. This file now keeps only
the CLI-driven run configuration and the training loop itself.

Usage:
    PYTHONNOUSERSITE=1 python scripts/train_stage1.py [N_SCENES]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import torch  # noqa: E402
from mmengine import Config  # noqa: E402

from loss import OPENOCC_LOSS  # noqa: E402

from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402
from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402
from src.datasets.occ4dgs_clip_dataset import Occ4DGSClipDataset  # noqa: E402

from src.training.stage_b_engine import (  # noqa: E402
    REPO_ROOT,
    GF3D_ROOT,
    HELDOUT_SCENES,
    USE_SPAWN_HEAD,
    build_pipeline,
    to_batch_of_one,
    to_cuda,
    build_stage_a,
    build_temporal_module,
    get_real_g0,
    deform_one_step,
    splat_and_loss,
    build_heldout_clip_datasets,
    evaluate_heldout,
    CurrentFrameEncoder,
    ReferenceBuffer,
)

import wandb  # noqa: E402

wandb.init(mode="disabled")


SEED = 42

ALL_TRAINED_SCENES = ["scene-0061", "scene-0103", "scene-0553", "scene-0655",
                      "scene-0757", "scene-0796", "scene-0916", "scene-1077"]
_n_scenes_arg = int(sys.argv[1]) if len(sys.argv) > 1 else len(ALL_TRAINED_SCENES)
SCENES = ALL_TRAINED_SCENES[:_n_scenes_arg]

N_EPOCHS = 20
EVAL_EVERY_OPT_STEPS = 20
GRAD_ACCUM_STEPS = 4
LR = 1e-4
WEIGHT_DECAY = 0.05
MOTION_PENALTY_WEIGHT = 0.01


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
          f"eval every {EVAL_EVERY_OPT_STEPS} OPTIMIZER STEPS, SpawnHead={'ON' if USE_SPAWN_HEAD else 'OFF'}")

    heldout_clip_datasets = build_heldout_clip_datasets(nusc, full_frame_index)
    print(f"Held-out (never trained): {HELDOUT_SCENES}, "
          f"{sum(len(d) for d in heldout_clip_datasets)} clips total")

    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_stage_a(cfg)
    encoder = CurrentFrameEncoder(segmentor)
    torch.manual_seed(SEED)
    pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout, spawn_head = build_temporal_module()
    loss_func_for_baseline = OPENOCC_LOSS.build(cfg.loss).cuda()

    heldout_donothing = evaluate_heldout(
        segmentor, encoder, pool, hypernet, deform_mu, deform_r,
        feature_dropout, z_dropout, cfg, loss_func_for_baseline,
        heldout_clip_datasets, mode="donothing", spawn_head=spawn_head
    )
    print(f"Held-out do-nothing baseline (adjusted mIoU, fixed reference): "
          f"{heldout_donothing:.3f}")

    trainable_params = (
        list(pool.parameters()) + list(hypernet.parameters())
        + list(deform_mu.parameters()) + list(deform_r.parameters())
    )
    if spawn_head is not None:
        trainable_params += list(spawn_head.parameters())
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
                feature_dropout, z_dropout, cuda0, cuda1,
                spawn_head=spawn_head, return_deltas=True
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
                        heldout_clip_datasets, mode="trained", spawn_head=spawn_head
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
                            "spawn_head": ({k: v.clone() for k, v in spawn_head.state_dict().items()}
                                           if spawn_head is not None else None),
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
        "spawn_head": spawn_head.state_dict() if spawn_head is not None else None,
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
            "spawn_head": best_state["spawn_head"],
            "trained_on_scenes": SCENES, "n_epochs": N_EPOCHS,
            "best_opt_step": best_state["opt_step"], "best_heldout_delta": best_state["heldout_delta"],
        }, best_ckpt_path)
        print(f"Saved BEST-held-out-delta temporal module (opt_step {best_state['opt_step']}, "
              f"delta={best_state['heldout_delta']:+.3f}) to {best_ckpt_path}")

    print(f"\nSUMMARY for SCENES={SCENES} ({len(SCENES)} scenes, {len(clip_dataset)} "
          f"clips/epoch, {opt_step_count} total optimizer steps, SpawnHead="
          f"{'ON' if USE_SPAWN_HEAD else 'OFF'}): best held-out delta = "
          f"{best_heldout_delta:+.3f} at opt_step "
          f"{best_state['opt_step'] if best_state else 'N/A'}")


if __name__ == "__main__":
    main()
