"""
scripts/evaluate_vggt_detailed.py

The training loop only ever prints ONE aggregate adjusted-mIoU number per checkpoint.
This breaks that open: (1) per-class IoU, using the trained checkpoint that just
finished (n=3 + dropout) -- does it fail uniformly across classes, or collapse on
specific ones the single aggregate number hides? (2) per-clip breakdown across the
full held-out set -- is underperformance spread evenly across all 78 held-out clips,
or concentrated in a handful of bad ones (which would look very different from
genuine, uniform underperformance)?

Run: PYTHONNOUSERSITE=1 python scripts/evaluate_vggt_detailed.py
(requires USE_VGGT_DEFORMABLE=True, and a saved checkpoint)
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
    REPO_ROOT, GF3D_ROOT, CLASS_NAMES, PC_RANGE, HELDOUT_SCENES,
    build_pipeline, to_batch_of_one, to_cuda,
    build_stage_a, build_temporal_module, get_real_g0, deform_one_step,
    splat_and_loss, adjusted_miou, USE_VGGT_DEFORMABLE,
)
from misc.metric_util import MeanIoU  # noqa: E402


def per_class_ious(miou_metric, class_names):
    """Like adjusted_miou(), but returns the per-class breakdown instead of
    collapsing to a single mean."""
    total_seen = miou_metric.total_seen[:-1]
    total_correct = miou_metric.total_correct[:-1]
    total_positive = miou_metric.total_positive[:-1]
    results = {}
    for i, name in enumerate(class_names):
        if total_seen[i].item() == 0:
            results[name] = None  # never appeared in this eval set at all
            continue
        union = (total_seen[i] + total_positive[i] - total_correct[i]).item()
        results[name] = (total_correct[i].item() / union * 100) if union > 0 else 0.0
    return results


def main():
    assert USE_VGGT_DEFORMABLE, "Set USE_VGGT_DEFORMABLE=True in stage_b_engine.py first."

    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        full_frame_index = json.load(f)

    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_stage_a(cfg)
    pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout, spawn_head = build_temporal_module()

    ckpt_path = os.path.join(
        REPO_ROOT, "experiments", "stage_b_temporal_checkpoints", "stage1_warmup_temporal_n3_final.pth"
    )
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cuda")
    pool.load_state_dict(ckpt["pool"])
    print(f"Loaded checkpoint: {ckpt_path}\n")
    for m in [pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout]:
        m.eval()

    from loss import OPENOCC_LOSS
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    # Build held-out clip datasets directly (same pattern as build_heldout_clip_datasets)
    heldout_clip_datasets = []
    for scene_name in HELDOUT_SCENES:
        frame_index = {scene_name: full_frame_index[scene_name]}
        base = Occ4DGSDataset(
            nusc, frame_index,
            os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
            os.path.join(REPO_ROOT, "data", "occ3d_gts"),
            pipeline=build_pipeline(),
        )
        heldout_clip_datasets.append((scene_name, Occ4DGSClipDataset(base, unroll_window=2)))

    overall_miou = MeanIoU(list(range(1, 17)), 17, CLASS_NAMES, True, 17, filter_minmax=False)
    overall_miou.reset()
    donothing_miou = MeanIoU(list(range(1, 17)), 17, CLASS_NAMES, True, 17, filter_minmax=False)
    donothing_miou.reset()

    per_clip_results = []  # (scene_name, clip_idx, trained_delta_vs_donothing)

    with torch.no_grad():
        for scene_name, clip_dataset in heldout_clip_datasets:
            for clip_idx in range(len(clip_dataset)):
                frame0_dict, frame1_dict = clip_dataset[clip_idx]
                cuda0 = to_cuda(to_batch_of_one(frame0_dict))
                cuda1 = to_cuda(to_batch_of_one(frame1_dict))
                g0, g0_dict = get_real_g0(segmentor, cuda0)

                g1 = deform_one_step(g0, None, pool, hypernet, deform_mu, deform_r,
                                      feature_dropout, z_dropout, cuda0, cuda1,
                                      spawn_head=spawn_head)
                _, _, head_out = splat_and_loss(segmentor, g1, type(g0_dict), cuda1, cfg, loss_func)
                gt_occ = head_out["sampled_label"][0]
                mask = head_out["occ_mask"].flatten(1)[0].bool()
                pred = head_out["pred_occ"][-1][0].argmax(0)

                clip_miou = MeanIoU(list(range(1, 17)), 17, CLASS_NAMES, True, 17, filter_minmax=False)
                clip_miou.reset()
                clip_miou._after_step(pred, gt_occ, mask)
                trained_score = adjusted_miou(clip_miou, CLASS_NAMES)

                overall_miou._after_step(pred, gt_occ, mask)

                _, _, do_head_out = splat_and_loss(segmentor, g0, type(g0_dict), cuda1, cfg, loss_func)
                do_pred = do_head_out["pred_occ"][-1][0].argmax(0)
                do_clip_miou = MeanIoU(list(range(1, 17)), 17, CLASS_NAMES, True, 17, filter_minmax=False)
                do_clip_miou.reset()
                do_clip_miou._after_step(do_pred, gt_occ, mask)
                donothing_score = adjusted_miou(do_clip_miou, CLASS_NAMES)
                donothing_miou._after_step(do_pred, gt_occ, mask)

                delta = trained_score - donothing_score
                per_clip_results.append((scene_name, clip_idx, trained_score, donothing_score, delta))

    print("=" * 80)
    print("PER-CLIP BREAKDOWN (trained vs do-nothing, this specific checkpoint)")
    print("=" * 80)
    for scene_name, clip_idx, trained_score, donothing_score, delta in per_clip_results:
        flag = "  <-- WORSE than do-nothing" if delta < 0 else ""
        print(f"  {scene_name} clip {clip_idx:3d}: trained={trained_score:6.3f}  "
              f"do-nothing={donothing_score:6.3f}  delta={delta:+6.3f}{flag}")

    deltas = [d for *_, d in per_clip_results]
    n_worse = sum(1 for d in deltas if d < 0)
    n_better = sum(1 for d in deltas if d > 0)
    print(f"\n{len(deltas)} total clips: {n_better} better than do-nothing, "
          f"{n_worse} worse, {len(deltas) - n_better - n_worse} tied")
    print(f"delta distribution: mean={sum(deltas)/len(deltas):.4f}, "
          f"min={min(deltas):.4f}, max={max(deltas):.4f}")
    print("(if a small number of clips are dragging the average down while most are "
          "fine, that's a very different story than uniform underperformance)")

    print()
    print("=" * 80)
    print("PER-CLASS IoU (aggregated over ALL held-out clips)")
    print("=" * 80)
    trained_per_class = per_class_ious(overall_miou, CLASS_NAMES)
    donothing_per_class = per_class_ious(donothing_miou, CLASS_NAMES)
    for name in CLASS_NAMES:
        t = trained_per_class[name]
        d = donothing_per_class[name]
        if t is None or d is None:
            print(f"  {name:20s}: never appeared in held-out set")
            continue
        diff = t - d
        flag = "  <-- WORSE than do-nothing" if diff < 0 else ""
        print(f"  {name:20s}: trained={t:6.2f}  do-nothing={d:6.2f}  diff={diff:+6.2f}{flag}")

    print(f"\nOverall aggregate (matches training log convention): "
          f"trained={adjusted_miou(overall_miou, CLASS_NAMES):.3f}, "
          f"do-nothing={adjusted_miou(donothing_miou, CLASS_NAMES):.3f}")

    print("\nDetailed evaluation complete.")


if __name__ == "__main__":
    main()
