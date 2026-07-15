"""
Phase 1, steps 3-4: build the per-scene frame index with has_gt tags, and analyze
contiguous valid-GT run lengths.
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.nuscenes_mini import load_nuscenes, build_all_scene_frames
from src.datasets.occ3d_gt import load_occ3d_labels

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUSCENES_ROOT = os.path.join(REPO_ROOT, "data", "nuscenes_mini")
OCC3D_ROOT = os.path.join(REPO_ROOT, "data", "occ3d_gts")
OUT_PATH = os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")


def contiguous_runs(flags):
    runs, current = [], 0
    for f in flags:
        if f:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return runs


def main():
    nusc = load_nuscenes(NUSCENES_ROOT)
    scene_frames = build_all_scene_frames(nusc)

    index = {}
    print(f"{'scene':<14}{'#frames':>10}{'#has_gt':>10}{'runs>=3':>10}{'max_run':>10}")
    for scene_name, tokens in scene_frames.items():
        entries = [{"sample_token": t, "has_gt": load_occ3d_labels(OCC3D_ROOT, scene_name, t) is not None}
                   for t in tokens]
        index[scene_name] = entries

        flags = [e["has_gt"] for e in entries]
        runs = contiguous_runs(flags)
        n_runs_ge3 = sum(1 for r in runs if r >= 3)
        max_run = max(runs) if runs else 0
        print(f"{scene_name:<14}{len(flags):>10}{sum(flags):>10}{n_runs_ge3:>10}{max_run:>10}")
        if not any(r >= 3 for r in runs):
            print(f"  !! WARNING: {scene_name} has NO contiguous valid-GT run >= 3 frames")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nWrote frame index to {OUT_PATH}")


if __name__ == "__main__":
    main()