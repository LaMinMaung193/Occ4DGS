"""
scripts/make_tiered_infos.py

Step 3 (full-dataset plan): creates scene-limited subsets of our already-generated
nuscenes_infos_gf3d_{train,val}.pkl, for the staged small/medium/full gate-check
strategy -- so each tier's held-out validation pass stays proportionally cheap too,
not just the training pass.

Confirmed necessary because: val_dataset_config always points at the FULL val
split (6,019 keyframes) regardless of train scene count -- train.py's automatic
per-epoch mIoU evaluation (confirmed: eval_every_epochs defaults to 1) would
otherwise dominate wall-clock time for a supposedly-cheap small-tier gate check.
Mirrors our own project's existing convention (Steps 1-4's small, fixed
HELDOUT_SCENES), applied here at full-dataset scale.

Reads existing .pkl files READ-ONLY ("rb"). Writes NEW files only, into our own
separate /media/user/1TSSD/min/gf3d_infos/ folder -- nothing existing is
overwritten, moved, or deleted.

Usage:
    PYTHONNOUSERSITE=1 python scripts/make_tiered_infos.py --tier small --n-train-scenes 50 --n-val-scenes 10
    PYTHONNOUSERSITE=1 python scripts/make_tiered_infos.py --tier medium --n-train-scenes 200 --n-val-scenes 30
"""
import argparse
import os
import pickle

INFOS_DIR = "/media/user/1TSSD/min/gf3d_infos"


def make_subset(split, n_scenes, tier_name, seed=42):
    src_path = os.path.join(INFOS_DIR, f"nuscenes_infos_gf3d_{split}.pkl")
    print(f"[{split}] reading {src_path}")
    with open(src_path, "rb") as f:
        data = pickle.load(f)

    all_scene_tokens = sorted(data["infos"].keys())
    print(f"[{split}] full set has {len(all_scene_tokens)} scenes, "
          f"{len(data['metadata'])} keyframes")

    if n_scenes >= len(all_scene_tokens):
        print(f"[{split}] requested {n_scenes} >= available {len(all_scene_tokens)}, "
              f"using all scenes (no reduction)")
        selected_scenes = set(all_scene_tokens)
    else:
        import random
        rng = random.Random(seed)
        selected_scenes = set(rng.sample(all_scene_tokens, n_scenes))

    new_infos = {k: v for k, v in data["infos"].items() if k in selected_scenes}
    new_metadata = [(scene, idx) for scene, idx in data["metadata"] if scene in selected_scenes]

    new_data = {"infos": new_infos, "metadata": new_metadata}

    dst_path = os.path.join(INFOS_DIR, f"nuscenes_infos_gf3d_{split}_{tier_name}.pkl")
    with open(dst_path, "wb") as f:
        pickle.dump(new_data, f)

    print(f"[{split}] wrote {len(new_infos)} scenes, {len(new_metadata)} keyframes "
          f"-> {dst_path}")
    return dst_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", required=True, help="tier name, e.g. small/medium")
    parser.add_argument("--n-train-scenes", type=int, required=True)
    parser.add_argument("--n-val-scenes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = make_subset("train", args.n_train_scenes, args.tier, args.seed)
    val_path = make_subset("val", args.n_val_scenes, args.tier, args.seed)

    print(f"\nDone. To use this tier, point a config's train_dataset_config/"
          f"val_dataset_config 'imageset' at:")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")


if __name__ == "__main__":
    main()
