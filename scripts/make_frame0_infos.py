"""
scripts/make_frame0_infos.py

Stage B prep: creates a single info file containing ONLY frame 0 (index==0) of
EVERY scene, combining both train and val splits -- since Stage A only ever runs
once per scene (frame 0), regardless of which split a scene belongs to. This is
what the G_0 extraction script (extract_g0_cache.py) iterates over.

Reads existing nuscenes_infos_gf3d_{train,val}.pkl READ-ONLY. Writes one new file
into our own separate folder -- nothing existing touched.

IMPORTANT: writes defensively (temp file + explicit flush/fsync + read-back
verification + atomic rename) -- confirmed necessary after this exact write
silently produced a corrupted file TWICE on this external SSD (once as a 0-byte
file, once as a plausible-sized but still unreadable file, both times with no
error reported at write time). Do not revert to a plain open()/dump() without
this safeguard for any file written to this drive.
"""
import os
import pickle

INFOS_DIR = "/media/user/1TSSD/min/gf3d_infos"
OUT_PATH = os.path.join(INFOS_DIR, "nuscenes_infos_gf3d_frame0_all.pkl")
TMP_PATH = OUT_PATH + ".tmp"


def main():
    combined_infos = {}
    combined_metadata = []

    for split in ("train", "val"):
        src_path = os.path.join(INFOS_DIR, f"nuscenes_infos_gf3d_{split}.pkl")
        print(f"[{split}] reading {src_path}")
        with open(src_path, "rb") as f:
            data = pickle.load(f)

        n_scenes_this_split = 0
        for scene_token, idx in data["metadata"]:
            if idx == 0:
                combined_infos[scene_token] = data["infos"][scene_token]
                combined_metadata.append((scene_token, idx))
                n_scenes_this_split += 1
        print(f"[{split}] kept {n_scenes_this_split} frame-0 entries "
              f"(out of {len(set(s for s, i in data['metadata']))} scenes)")

    print(f"\nCombined: {len(combined_infos)} scenes, "
          f"{len(combined_metadata)} frame-0 entries total (expect 850)")

    out_data = {"infos": combined_infos, "metadata": combined_metadata}

    print(f"Writing to temp file: {TMP_PATH}")
    with open(TMP_PATH, "wb") as f:
        pickle.dump(out_data, f)
        f.flush()
        os.fsync(f.fileno())  # force the OS to actually commit to disk, not just
                              # buffer it -- confirmed necessary given repeated
                              # silent write corruption on this drive otherwise

    print("Verifying by reading the temp file back...")
    with open(TMP_PATH, "rb") as f:
        verify_data = pickle.load(f)
    assert len(verify_data["infos"]) == len(combined_infos), "infos count mismatch on verify"
    assert len(verify_data["metadata"]) == len(combined_metadata), "metadata count mismatch on verify"
    print(f"  Verified OK: {len(verify_data['infos'])} scenes, "
          f"{len(verify_data['metadata'])} entries read back correctly")

    os.replace(TMP_PATH, OUT_PATH)  # atomic rename -- only now does the real
                                     # target filename exist, and only with
                                     # confirmed-valid content
    print(f"Wrote and verified -> {OUT_PATH}")


if __name__ == "__main__":
    main()
