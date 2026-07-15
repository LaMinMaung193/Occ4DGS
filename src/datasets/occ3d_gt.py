"""
Loads Occ3D-nuScenes voxel GT for a given sample token.

Confirmed file structure (Phase 0/1, EXPERIMENT_LOG.md): this GT dump uses TWO
different on-disk conventions depending on scene:
    (a) data/occ3d_gts/<scene-name>/<sample-token>/labels.npz  (subfolder + labels.npz)
    (b) data/occ3d_gts/<scene-name>/<sample-token>.npz          (flat file)
Both must be checked -- do not assume one convention across all scenes.

Each labels.npz/*.npz contains THREE arrays: semantics (200,200,16) uint8 0-17,
mask_lidar (200,200,16) uint8 binary, mask_camera (200,200,16) uint8 binary.
"""
import os
import numpy as np


def load_occ3d_labels(occ3d_gts_root, scene_name, sample_token):
    """Returns dict {semantics, mask_lidar, mask_camera}, or None if no GT exists."""
    candidates = [
        os.path.join(occ3d_gts_root, scene_name, sample_token, "labels.npz"),
        os.path.join(occ3d_gts_root, scene_name, sample_token + ".npz"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None
    data = np.load(path)
    return {
        "semantics": data["semantics"],
        "mask_lidar": data["mask_lidar"],
        "mask_camera": data["mask_camera"],
    }