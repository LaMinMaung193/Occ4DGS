"""
scripts/build_stageb_manifest.py

Stage B training prep -- rebuilt to keep train and val GENUINELY separate
(the original version incorrectly combined both splits into one pool with no
held-out set at all -- caught before any real training run, fixed here).

For every scene in each split with a cached G_0, finds the real frame 0 ->
next genuinely-moving keyframe pair (>=0.5m real ego translation). Writes
separate manifests + paired info pkls for train and val, so val scenes are
NEVER touched during training -- only used for held-out evaluation.

Written defensively (temp file + fsync + read-back verification) given this
drive's own confirmed history of silent write corruption.

Run from Occ4DGS repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/build_stageb_manifest.py
"""
import json
import os
import pickle

TRAIN_PKL = "/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_train.pkl"
VAL_PKL = "/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_val.pkl"
G0_CACHE_DIR = "/media/user/1TSSD/min/g0_cache"
OUT_DIR = "/media/user/1TSSD/min/stageb_training"
MIN_TRANSLATION_M = 0.5


def find_next_keyframe_index(scene_infos, after_index=0):
    for i in range(after_index + 1, len(scene_infos)):
        if scene_infos[i].get("is_key_frame"):
            return i
    return None


def find_next_moving_keyframe_index(scene_infos, after_index, min_translation_m=MIN_TRANSLATION_M):
    idx = after_index
    while True:
        next_idx = find_next_keyframe_index(scene_infos, after_index=idx)
        if next_idx is None:
            return None, None
        t0 = scene_infos[after_index]["data"]["LIDAR_TOP"]["pose"]["translation"]
        t1 = scene_infos[next_idx]["data"]["LIDAR_TOP"]["pose"]["translation"]
        delta = sum((a - b) ** 2 for a, b in zip(t0, t1)) ** 0.5
        if delta >= min_translation_m:
            return next_idx, delta
        idx = next_idx


def build_split(split_name, infos_pkl_path, cached_scenes, out_dir):
    print(f"\n=== {split_name} split ===")
    with open(infos_pkl_path, "rb") as f:
        data = pickle.load(f)
    all_infos = data["infos"]
    print(f"  {len(all_infos)} scenes in this split")

    manifest = {}
    new_infos = {}
    new_metadata = []
    n_no_g0 = 0
    n_no_moving_pair = 0
    translations = []

    for scene_token, scene_infos in all_infos.items():
        if scene_token not in cached_scenes:
            n_no_g0 += 1
            continue
        next_idx, delta = find_next_moving_keyframe_index(scene_infos, after_index=0)
        if next_idx is None:
            n_no_moving_pair += 1
            continue
        manifest[scene_token] = {"next_idx": next_idx, "translation_m": delta}
        new_infos[scene_token] = scene_infos
        new_metadata.append((scene_token, 0))
        new_metadata.append((scene_token, next_idx))
        translations.append(delta)

    n_valid = len(manifest)
    print(f"  Valid: {n_valid} | skipped (no G_0): {n_no_g0} | "
          f"skipped (no moving pair): {n_no_moving_pair}")
    if translations:
        translations.sort()
        n = len(translations)
        print(f"  Translation: min={translations[0]:.3f}m, median={translations[n//2]:.3f}m, "
              f"max={translations[-1]:.3f}m")

    manifest_path = os.path.join(out_dir, f"stageb_manifest_{split_name}.json")
    pairs_pkl_path = os.path.join(out_dir, f"nuscenes_infos_gf3d_stageb_pairs_{split_name}.pkl")

    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    with open(tmp) as f:
        verify = json.load(f)
    assert len(verify) == n_valid
    os.replace(tmp, manifest_path)

    out_data = {"infos": new_infos, "metadata": new_metadata}
    tmp = pairs_pkl_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out_data, f)
        f.flush()
        os.fsync(f.fileno())
    with open(tmp, "rb") as f:
        verify = pickle.load(f)
    assert len(verify["metadata"]) == n_valid * 2
    os.replace(tmp, pairs_pkl_path)

    print(f"  Written and verified: {manifest_path}, {pairs_pkl_path}")
    return n_valid


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cached_scenes = {f[:-3] for f in os.listdir(G0_CACHE_DIR) if f.endswith(".pt")}
    print(f"Scenes with cached G_0: {len(cached_scenes)}")

    n_train = build_split("train", TRAIN_PKL, cached_scenes, OUT_DIR)
    n_val = build_split("val", VAL_PKL, cached_scenes, OUT_DIR)

    print(f"\n{'='*60}")
    print(f"DONE. Train: {n_train} scenes. Val (held-out): {n_val} scenes.")
    print(f"Val scenes are NEVER used in training -- evaluation only.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
