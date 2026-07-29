"""
scripts/evaluate_stage_a_all_scenes.py

Diagnostic (Decision 1, confound isolation): evaluates Stage A's OWN per-frame
occupancy reconstruction quality -- no Stage B, no deformation, no motion at all --
across all 10 mini scenes, using the existing stage_a_best.pth checkpoint UNMODIFIED
(trained only on scene-0061, 60 epochs). No training happens in this script.

Purpose: isolate whether the held-out-scene generalization failure (scene-0103) was
(a) Stage B's motion prediction not generalizing, or (b) G_0 itself already being
degraded on any scene Stage A hasn't seen, confounding (a). This script only tests (b),
in isolation, by removing Stage B from the picture entirely.

Also reports an "adjusted mIoU" per scene that excludes zero-ground-truth-support
classes from the average, to avoid the same misleading-100%-IoU artifact that inflated
scene-0103's raw mIoU in the held-out comparison (barrier/bus/construction_vehicle had
zero GT voxels there and defaulted to 100%). This adjustment is a straightforward
intersection-over-union recomputation from MeanIoU's own accumulated
total_seen/total_correct/total_positive tensors, not a change to the reused MeanIoU
class itself -- flagged as an approximation of its internal formula, not independently
confirmed against MeanIoU's source, since the standard IoU formula is unambiguous
regardless.

Run from repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/evaluate_stage_a_all_scenes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from mmengine import Config  # noqa: E402

from train_stage1 import (  # noqa: E402 -- reusing exact helper functions
    build_stage_a,
    get_real_g0,
    splat_and_loss,
    to_cuda,
    CLASS_NAMES,
    GF3D_ROOT,
    REPO_ROOT,
)
import model  # noqa: E402,F401
from loss import OPENOCC_LOSS  # noqa: E402
from misc.metric_util import MeanIoU  # noqa: E402

from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402
from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402

import json  # noqa: E402

from run_stage_a_frame0 import build_pipeline, to_batch_of_one  # noqa: E402


ALL_SCENES = [
    "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100",
]
TRAINED_ON = ["scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757", "scene-0796", "scene-0916", "scene-1077"]  # keep in sync with train_stage_a.py's SCENES -- scene-1094/scene-1100 are the permanent held-out validation pair


def adjusted_miou(miou_metric, class_names):
    """Recompute mean IoU excluding classes with zero GT support in this scene,
    from the metric's own accumulated tensors, rather than trusting the raw average
    (which silently includes 0/0-defaults-to-100% classes)."""
    total_seen = miou_metric.total_seen[:-1]      # drop the summary "all classes" slot
    total_correct = miou_metric.total_correct[:-1]
    total_positive = miou_metric.total_positive[:-1]
    ious, kept_names, dropped_names = [], [], []
    for i, name in enumerate(class_names):
        if total_seen[i].item() == 0:
            dropped_names.append(name)
            continue
        union = (total_seen[i] + total_positive[i] - total_correct[i]).item()
        iou = (total_correct[i].item() / union) if union > 0 else 0.0
        ious.append(iou)
        kept_names.append(name)
    adjusted = (sum(ious) / len(ious) * 100) if ious else float("nan")
    return adjusted, kept_names, dropped_names


def main():
    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        full_frame_index = json.load(f)

    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_stage_a(cfg)  # loads stage_a_best.pth, frozen+eval
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    results = {}
    with torch.no_grad():
        for scene_name in ALL_SCENES:
            frame_index = {scene_name: full_frame_index[scene_name]}
            dataset = Occ4DGSDataset(
                nusc, frame_index,
                os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
                os.path.join(REPO_ROOT, "data", "occ3d_gts"),
                pipeline=build_pipeline(),
            )

            miou_metric = MeanIoU(list(range(1, 17)), 17, CLASS_NAMES, True, 17,
                                   filter_minmax=False)
            miou_metric.reset()

            for idx in range(len(dataset)):
                sample = dataset[idx]
                cuda_frame = to_cuda(to_batch_of_one(sample))
                g0, g0_dict = get_real_g0(segmentor, cuda_frame)
                _, _, head_out = splat_and_loss(
                    segmentor, g0, type(g0_dict), cuda_frame, cfg, loss_func
                )
                gt_occ = head_out["sampled_label"][0]
                mask = head_out["occ_mask"].flatten(1)[0].bool()
                pred = head_out["pred_occ"][-1][0].argmax(0)
                miou_metric._after_step(pred, gt_occ, mask)

            raw_miou, raw_iou2 = miou_metric._after_epoch()
            adj_miou, kept, dropped = adjusted_miou(miou_metric, CLASS_NAMES)

            tag = " (TRAINED ON)" if scene_name in TRAINED_ON else " (unseen)"
            print(f"\n=== {scene_name}{tag}: {len(dataset)} frames ===")
            print(f"  raw mIoU={raw_miou:.3f}  iou2={raw_iou2:.3f}")
            print(f"  adjusted mIoU (excl. {len(dropped)} zero-support classes: "
                  f"{dropped})={adj_miou:.3f}  over {len(kept)} classes: {kept}")

            results[scene_name] = {
                "raw_miou": raw_miou, "raw_iou2": raw_iou2,
                "adjusted_miou": adj_miou, "n_dropped_classes": len(dropped),
            }

    print("\n\n=== SUMMARY (adjusted mIoU, excludes zero-support classes) ===")
    for scene_name in ALL_SCENES:
        tag = " (TRAINED ON)" if scene_name in TRAINED_ON else ""
        r = results[scene_name]
        print(f"  {scene_name:14s}{tag:14s}adjusted_mIoU={r['adjusted_miou']:6.3f}  "
              f"raw_mIoU={r['raw_miou']:6.3f}  dropped_classes={r['n_dropped_classes']}")


if __name__ == "__main__":
    main()