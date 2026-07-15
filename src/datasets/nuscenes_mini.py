"""
Loads the 10 nuScenes v1.0-mini scenes via nuscenes-devkit and returns ordered per-scene
frame lists (sample tokens + sensor file paths).

See docs/IMPLEMENTATION_ROADMAP.md Phase 1, step 1.
"""
import os
from nuscenes.nuscenes import NuScenes

CAM_NAMES = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT",
]
LIDAR_NAME = "LIDAR_TOP"


def load_nuscenes(dataroot):
    return NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)


def get_scene_sample_tokens(nusc, scene_name):
    """Ordered list of sample tokens for one scene, following first_sample_token -> next."""
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    tokens = []
    token = scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = nusc.get("sample", token)["next"]
    return tokens


def get_sample_sensor_paths(nusc, sample_token):
    """Dict of {sensor_name: absolute_file_path} for one sample (6 cams + LiDAR)."""
    sample = nusc.get("sample", sample_token)
    paths = {}
    for cam in CAM_NAMES:
        sd = nusc.get("sample_data", sample["data"][cam])
        paths[cam] = os.path.join(nusc.dataroot, sd["filename"])
    lidar_sd = nusc.get("sample_data", sample["data"][LIDAR_NAME])
    paths[LIDAR_NAME] = os.path.join(nusc.dataroot, lidar_sd["filename"])
    return paths


def build_all_scene_frames(nusc):
    """{scene_name: [sample_token, ...]} for all 10 mini scenes, in temporal order."""
    return {s["name"]: get_scene_sample_tokens(nusc, s["name"]) for s in nusc.scene}