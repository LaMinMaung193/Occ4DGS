"""
scripts/build_stageb_manifest.py

Stage B training prep: for every one of the 850 scenes we have a cached G_0
for, find the real "frame 0 -> next genuinely-moving keyframe" pair -- the
same logic check_cross_frame.py validated for one scene, generalized across
all of them.

Builds two outputs:
  1. A manifest (scene_token -> {next_idx, translation_m}) for scenes where a
     valid pair was found -- some scenes may not have one (e.g. very short
     scenes, or scenes with no keyframe ever exceeding the motion threshold).
  2. A new info pkl containing BOTH frame 0 and the found next_idx for every
     valid scene, in GF3D's real (scene_token, index) format -- lets us reuse
     GaussianFormer3D's own real NuScenesDataset for loading, exactly like
     check_cross_frame.py already does, just across all scenes instead of one.

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
OUT_MANIFEST = "/media/user/1TSSD/min/gf3d_infos/stageb_manifest.json"
OUT_INFOS_PKL = "/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_stageb_pairs.pkl"
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


def main():
    print("Loading train + val info files...")
    with open(TRAIN_PKL, "rb") as f:
        train_data = pickle.load(f)
    with open(VAL_PKL, "rb") as f:
        val_data = pickle.load(f)

    all_infos = {}
    all_infos.update(train_data["infos"])
    all_infos.update(val_data["infos"])
    print(f"  Combined: {len(all_infos)} scenes total")

    cached_scenes = {f[:-3] for f in os.listdir(G0_CACHE_DIR) if f.endswith(".pt")}
    print(f"  Scenes with cached G_0: {len(cached_scenes)}")

    manifest = {}
    new_infos = {}
    new_metadata = []

    n_no_g0 = 0
    n_no_moving_pair = 0
    n_valid = 0
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
        n_valid += 1
        translations.append(delta)

    print(f"\n=== Results ===")
    print(f"  Valid scenes (G_0 cached + moving pair found): {n_valid}")
    print(f"  Skipped (no cached G_0): {n_no_g0}")
    print(f"  Skipped (no moving keyframe pair found): {n_no_moving_pair}")
    if translations:
        translations.sort()
        n = len(translations)
        print(f"  Translation distribution: min={translations[0]:.3f}m, "
              f"median={translations[n//2]:.3f}m, max={translations[-1]:.3f}m, "
              f"mean={sum(translations)/n:.3f}m")

    print(f"\nWriting manifest -> {OUT_MANIFEST}")
    tmp_manifest = OUT_MANIFEST + ".tmp"
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    with open(tmp_manifest) as f:
        verify_manifest = json.load(f)
    assert len(verify_manifest) == n_valid
    os.replace(tmp_manifest, OUT_MANIFEST)
    print(f"  Verified: {len(verify_manifest)} entries read back correctly")

    print(f"\nWriting paired info pkl -> {OUT_INFOS_PKL}")
    out_data = {"infos": new_infos, "metadata": new_metadata}
    tmp_pkl = OUT_INFOS_PKL + ".tmp"
    with open(tmp_pkl, "wb") as f:
        pickle.dump(out_data, f)
        f.flush()
        os.fsync(f.fileno())
    with open(tmp_pkl, "rb") as f:
        verify_pkl = pickle.load(f)
    assert len(verify_pkl["metadata"]) == n_valid * 2
    os.replace(tmp_pkl, OUT_INFOS_PKL)
    print(f"  Verified: {len(verify_pkl['metadata'])} entries "
          f"({len(verify_pkl['infos'])} scenes x 2) read back correctly")

    print(f"\nDone. {n_valid} scenes ready for Stage B training.")


if __name__ == "__main__":
    main()
