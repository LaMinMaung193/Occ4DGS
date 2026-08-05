"""
scripts/measure_gt_motion.py

Tier 1 item 3: measures how much ground-truth occupancy actually differs between
adjacent frames (frame t vs frame t+1), independent of Stage A or Stage B entirely.
occ_label is ego-centric per frame (LoadOccupancyOcc3d's use_ego=True) -- comparing the
same voxel indices across two frames captures the TOTAL apparent motion (ego motion +
any independently-moving object's motion combined), which is exactly the quantity
Stage B's Delta_mu/Delta_r are being asked to predict.

Purpose: if very few voxels actually change label between adjacent frames, that
indicates limited learnable signal in this task at this frame gap (dt ~0.5s), an
explanation for the observed difficulty independent of data volume or architecture.
If many voxels change, that argues against "not enough signal" and keeps the
data-scale/architecture explanations as the more likely ones.

Run from repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/measure_gt_motion.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_cwd = os.getcwd()
from src.datasets.gf3d_pipeline import build_pipeline, to_batch_of_one  # noqa: E402
os.chdir(_cwd)

import torch  # noqa: E402
from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402
from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402
from src.datasets.occ4dgs_clip_dataset import Occ4DGSClipDataset  # noqa: E402

import json  # noqa: E402


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENES_TO_CHECK = ["scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
                   "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100"]
TRAINED_SCENES = set(SCENES_TO_CHECK[:8])

EMPTY_LABEL = 17
# Label values inferred from MeanIoU's own construction elsewhere in this repo:
# MeanIoU(list(range(1, 17)), 17, CLASS_NAMES, ...) -- class_indices=[1..16] positionally
# matched to CLASS_NAMES, so CLASS_NAMES[i] <-> label value (i+1), NOT i. Label 0 is
# excluded from class_indices entirely (likely an unused/"noise" class), 17 is empty.
# FLAGGED: inferred from the constructor call pattern, not independently re-verified
# against Occ3D-nuScenes' own label definition file.
#   1=barrier, 2=bicycle, 3=bus, 4=car, 5=construction_vehicle, 6=motorcycle,
#   7=pedestrian, 8=traffic_cone, 9=trailer, 10=truck, 11=driveable_surface,
#   12=other_flat, 13=sidewalk, 14=terrain, 15=manmade, 16=vegetation
STATIC_LABELS = {1, 8, 11, 12, 13, 14, 15, 16}
DYNAMIC_LABELS = {2, 3, 4, 5, 6, 7, 9, 10}


def main():
    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        full_frame_index = json.load(f)

    pipeline = build_pipeline()
    overall_changed_frac = []
    overall_static_changed_frac = []
    overall_dynamic_changed_frac = []
    per_scene_results = {}

    for scene_name in SCENES_TO_CHECK:
        frame_index = {scene_name: full_frame_index[scene_name]}
        base = Occ4DGSDataset(
            nusc, frame_index,
            os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
            os.path.join(REPO_ROOT, "data", "occ3d_gts"),
            pipeline=pipeline,
        )
        clip_dataset = Occ4DGSClipDataset(base, unroll_window=2)

        scene_changed = []
        scene_static_changed = []
        scene_dynamic_changed = []

        for clip_idx in range(len(clip_dataset)):
            frame0_dict, frame1_dict = clip_dataset[clip_idx]
            batch0 = to_batch_of_one(frame0_dict)
            batch1 = to_batch_of_one(frame1_dict)

            occ0 = batch0["metas"]["occ_label"].flatten()
            occ1 = batch1["metas"]["occ_label"].flatten()
            assert occ0.shape == occ1.shape, "occ_label shape mismatch between frames"

            non_empty_either = (occ0 != EMPTY_LABEL) | (occ1 != EMPTY_LABEL)
            n_relevant = non_empty_either.sum().item()
            if n_relevant == 0:
                continue
            changed = (occ0 != occ1) & non_empty_either
            frac_changed = changed.sum().item() / n_relevant
            scene_changed.append(frac_changed)

            static_mask = torch.zeros_like(occ0, dtype=torch.bool)
            dynamic_mask = torch.zeros_like(occ0, dtype=torch.bool)
            for lbl in STATIC_LABELS:
                static_mask |= (occ0 == lbl) | (occ1 == lbl)
            for lbl in DYNAMIC_LABELS:
                dynamic_mask |= (occ0 == lbl) | (occ1 == lbl)

            if static_mask.sum().item() > 0:
                frac_static_changed = ((occ0 != occ1) & static_mask).sum().item() / static_mask.sum().item()
                scene_static_changed.append(frac_static_changed)
            if dynamic_mask.sum().item() > 0:
                frac_dynamic_changed = ((occ0 != occ1) & dynamic_mask).sum().item() / dynamic_mask.sum().item()
                scene_dynamic_changed.append(frac_dynamic_changed)

        mean_changed = sum(scene_changed) / len(scene_changed) if scene_changed else float("nan")
        mean_static = sum(scene_static_changed) / len(scene_static_changed) if scene_static_changed else float("nan")
        mean_dynamic = sum(scene_dynamic_changed) / len(scene_dynamic_changed) if scene_dynamic_changed else float("nan")

        tag = "(TRAINED)" if scene_name in TRAINED_SCENES else "(HELD-OUT)"
        print(f"{scene_name} {tag}: {len(clip_dataset)} clips  "
              f"mean_frac_changed(overall)={mean_changed:.4f}  "
              f"static={mean_static:.4f}  dynamic={mean_dynamic:.4f}")

        per_scene_results[scene_name] = (mean_changed, mean_static, mean_dynamic)
        overall_changed_frac.extend(scene_changed)
        overall_static_changed_frac.extend(scene_static_changed)
        overall_dynamic_changed_frac.extend(scene_dynamic_changed)

    print("\n=== OVERALL (all scenes, all clips) ===")
    print(f"Mean fraction of non-empty voxels that change label between adjacent "
          f"frames: {sum(overall_changed_frac)/len(overall_changed_frac):.4f}")
    print(f"  static-class voxels:  {sum(overall_static_changed_frac)/len(overall_static_changed_frac):.4f}")
    print(f"  dynamic-class voxels: {sum(overall_dynamic_changed_frac)/len(overall_dynamic_changed_frac):.4f}")

    trained_changed = [per_scene_results[s][0] for s in SCENES_TO_CHECK if s in TRAINED_SCENES]
    heldout_changed = [per_scene_results[s][0] for s in SCENES_TO_CHECK if s not in TRAINED_SCENES]
    print(f"\nTrained scenes mean overall change: {sum(trained_changed)/len(trained_changed):.4f}")
    print(f"Held-out scenes mean overall change: {sum(heldout_changed)/len(heldout_changed):.4f}")


if __name__ == "__main__":
    main()
