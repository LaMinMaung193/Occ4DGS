"""
scripts/diagnose_vggt_attn_gradient.py

Directly inspects gradients flowing into attn_mlp (vs. offset_mlp/query_update_mlp,
for comparison) on real data, to distinguish two very different explanations for
attn_mlp.2's weight/bias staying at exact zero-init across all 4 blocks after 585
real training steps (found via diagnose_vggt_param_coverage.py):
  (a) a genuine bug -- gradient is None or exactly 0.0 every step, nothing ever
      reaches attn_mlp at all -- the K=4 sampled values are staying degenerate
      (effectively identical) throughout training, not just at init as originally
      reasoned.
  (b) real but very small gradient -- attn_mlp IS learning, just so slowly that
      585 steps at this LR isn't enough to move it off exactly 0.0 in float32 --
      not a bug, just a slow-moving parameter.

A before/after weight comparison (diagnose_vggt_param_coverage.py) cannot
distinguish these; only inspecting .grad directly can.

Run: PYTHONNOUSERSITE=1 python scripts/diagnose_vggt_attn_gradient.py
(requires USE_VGGT_DEFORMABLE=True)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import torch  # noqa: E402
from mmengine import Config  # noqa: E402

from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402
from src.datasets.occ4dgs_clip_dataset import Occ4DGSClipDataset  # noqa: E402
from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402

from src.training.stage_b_engine import (  # noqa: E402
    REPO_ROOT, GF3D_ROOT, build_pipeline, to_batch_of_one, to_cuda,
    build_stage_a, build_temporal_module, get_real_g0, deform_one_step,
    splat_and_loss, USE_VGGT_DEFORMABLE,
)


def report_grad(name, param):
    if param.grad is None:
        print(f"  {name}: grad is None (NEVER reached by backward -- graph disconnected)")
        return
    g = param.grad
    print(f"  {name}: grad norm={g.norm().item():.3e}, "
          f"max abs={g.abs().max().item():.3e}, "
          f"nonzero elements={torch.count_nonzero(g).item()}/{g.numel()}")


def main():
    assert USE_VGGT_DEFORMABLE, "Set USE_VGGT_DEFORMABLE=True in stage_b_engine.py first."

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

    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_stage_a(cfg)
    pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout, spawn_head = build_temporal_module()

    # LOAD THE REAL TRAINED CHECKPOINT -- we want to inspect gradient behavior at
    # the model's ACTUAL current (post-training) state, not a fresh one, since the
    # question is "is this still stuck after training got this model to where it is."
    ckpt_path = os.path.join(
        REPO_ROOT, "experiments", "stage_b_temporal_checkpoints", "stage1_warmup_temporal_n3_final.pth"
    )
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cuda")
        pool.load_state_dict(ckpt["pool"])
        print(f"Loaded trained checkpoint from {ckpt_path}\n")
    else:
        print("No trained checkpoint found -- using fresh init instead.\n")

    from loss import OPENOCC_LOSS
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    g0, g0_dict = get_real_g0(segmentor, cuda0)
    pool.zero_grad()

    g1, delta_mu, delta_r = deform_one_step(
        g0, None, pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout,
        cuda0, cuda1, spawn_head=spawn_head, return_deltas=True,
    )
    loss, loss_dict, head_out = splat_and_loss(segmentor, g1, type(g0_dict), cuda1, cfg, loss_func)
    motion_penalty = delta_mu.pow(2).mean() + (1.0 - delta_r[..., 0].abs()).mean()
    total_loss = loss + 0.01 * motion_penalty
    print(f"task loss: {loss.item():.4f}, motion_penalty: {motion_penalty.item():.6f}, "
          f"total: {total_loss.item():.4f}\n")
    total_loss.backward()

    print("=" * 70)
    print("Gradient inspection, per block (real data, one real backward pass)")
    print("=" * 70)
    for i, block in enumerate(pool.blocks):
        print(f"\n--- Block {i} ---")
        report_grad("offset_mlp[0].weight (hidden layer)", block.offset_mlp[0].weight)
        report_grad("offset_mlp[2].weight (output layer)", block.offset_mlp[2].weight)
        report_grad("attn_mlp[0].weight   (hidden layer)", block.attn_mlp[0].weight)
        report_grad("attn_mlp[2].weight   (output layer)", block.attn_mlp[2].weight)
        report_grad("query_update_mlp[2].weight (output)", block.query_update_mlp[2].weight)

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
