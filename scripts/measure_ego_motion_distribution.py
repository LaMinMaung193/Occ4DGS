"""
scripts/measure_ego_motion_distribution.py

Measures real ego-motion magnitude (translation norm, rotation angle) across every
clip in ALL 10 mini-set scenes -- not just the 2 held-out scenes checked earlier ad
hoc. Purpose: decide whether the Gaussian-representational gap (fixed 6400 Gaussians
carried forward; large ego motion introduces new content with nothing representing
it -- EXPERIMENT_LOG.md) is a rare edge case or the common case at this frame rate,
before committing engineering time to a recycling/spawning fix.

Run from repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/measure_ego_motion_distribution.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import train_stage1 as ts1  # noqa: E402

from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402
from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402
from src.datasets.occ4dgs_clip_dataset import Occ4DGSClipDataset  # noqa: E402

import json  # noqa: E402


ALL_SCENES = ["scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
              "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100"]

BUCKETS = [0.5, 1.0, 2.0, 3.0, 5.0, float("inf")]


def bucket_label(norm, buckets=BUCKETS):
    prev = 0.0
    for b in buckets:
        if norm <= b:
            if b == float("inf"):
                return f">{prev}m"
            return f"{prev}-{b}m"
        prev = b


def main():
    nusc = load_nuscenes(os.path.join(ts1.REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(ts1.REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        full_frame_index = json.load(f)

    all_norms = []
    all_angles = []
    bucket_counts = {}
    per_scene_stats = {}

    for scene_name in ALL_SCENES:
        frame_index = {scene_name: full_frame_index[scene_name]}
        base = Occ4DGSDataset(
            nusc, frame_index,
            os.path.join(ts1.REPO_ROOT, "data", "nuscenes_mini"),
            os.path.join(ts1.REPO_ROOT, "data", "occ3d_gts"),
            pipeline=ts1.build_pipeline(),
        )
        clip_dataset = Occ4DGSClipDataset(base, unroll_window=2)

        scene_norms = []
        for clip_idx in range(len(clip_dataset)):
            frame0_dict, frame1_dict = clip_dataset[clip_idx]
            cuda0 = ts1.to_cuda(ts1.to_batch_of_one(frame0_dict))
            cuda1 = ts1.to_cuda(ts1.to_batch_of_one(frame1_dict))
            pose_prev = cuda0["metas"]["lidar2global"][0]
            pose_curr = cuda1["metas"]["lidar2global"][0]
            T = ts1.compute_relative_transform(pose_prev, pose_curr)
            norm = T[:3, 3].norm().item()

            import torch
            R = T[:3, :3]
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            angle_rad = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)).item()
            angle_deg = angle_rad * 180.0 / 3.14159265

            scene_norms.append(norm)
            all_norms.append(norm)
            all_angles.append(angle_deg)
            label = bucket_label(norm)
            bucket_counts[label] = bucket_counts.get(label, 0) + 1

        per_scene_stats[scene_name] = {
            "n_clips": len(clip_dataset),
            "min": min(scene_norms) if scene_norms else float("nan"),
            "max": max(scene_norms) if scene_norms else float("nan"),
            "mean": sum(scene_norms) / len(scene_norms) if scene_norms else float("nan"),
        }
        s = per_scene_stats[scene_name]
        print(f"{scene_name}: {s['n_clips']} clips  "
              f"min={s['min']:.3f}m  max={s['max']:.3f}m  mean={s['mean']:.3f}m")

    print(f"\n=== OVERALL ({len(all_norms)} clips across {len(ALL_SCENES)} scenes) ===")
    print(f"translation: min={min(all_norms):.3f}m  max={max(all_norms):.3f}m  "
          f"mean={sum(all_norms)/len(all_norms):.3f}m")
    print(f"rotation:    min={min(all_angles):.3f}deg  max={max(all_angles):.3f}deg  "
          f"mean={sum(all_angles)/len(all_angles):.3f}deg")

    sorted_norms = sorted(all_norms)
    n = len(sorted_norms)
    for pct in (10, 25, 50, 75, 90, 95):
        idx = min(int(n * pct / 100), n - 1)
        print(f"  p{pct}: {sorted_norms[idx]:.3f}m")

    print("\n=== DISTRIBUTION BY BUCKET (translation magnitude) ===")
    ordered_labels = []
    prev = 0.0
    for b in BUCKETS:
        ordered_labels.append(f"{prev}-{b}m" if b != float("inf") else f">{prev}m")
        prev = b
    for label in ordered_labels:
        count = bucket_counts.get(label, 0)
        pct = 100.0 * count / len(all_norms)
        print(f"  {label:>12s}: {count:4d} clips ({pct:5.1f}%)")

    high_motion = sum(1 for n_ in all_norms if n_ > 3.0)
    print(f"\nClips with translation > 3.0m (roughly where scene-1094 showed real "
          f"damage): {high_motion}/{len(all_norms)} ({100.0*high_motion/len(all_norms):.1f}%)")


if __name__ == "__main__":
    main()
