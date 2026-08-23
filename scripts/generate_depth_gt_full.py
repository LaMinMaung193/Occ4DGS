"""
scripts/generate_depth_gt_full.py

Step 2 (full-dataset plan, EXPERIMENT_LOG.md): full-scale adaptation of
generate_depth_gt.py (originally built for the 10-scene v1.0-mini dataset), for the
real v1.0-trainval nuScenes data now available.

Confirmed before writing this (not assumed): build_all_scene_frames(),
get_sample_sensor_paths(), CAM_NAMES, LIDAR_NAME are all genuinely general -- they
operate on nusc.scene/nusc.get(...) directly, no hardcoded mini-specific scene
names or counts (the "10 mini scenes" docstring was stale comment text, not real
logic). ONLY load_nuscenes()'s hardcoded version="v1.0-mini" needed changing -- done
here by constructing NuScenes(version="v1.0-trainval", ...) directly, WITHOUT
modifying the shared nuscenes_mini.py module (keeps this change fully contained to
this one script, doesn't risk the existing mini-dataset pipeline).

_sensor2global()/project_lidar_to_camera() (the actual projection math) are reused
COMPLETELY UNCHANGED -- already proven correct, no reason to touch.

Output goes to a SEPARATE folder (/media/user/1TSSD/min/depth_gt/), not directly
into Ruby's data/nuscenes/ -- per explicit instruction to keep our own generated
outputs clearly separated. Making this discoverable at the path
LoadMultiViewDepthFromFiles actually expects is a symlink step, deferred (along with
the anno_root wiring) to the end of Step 2, not done here.

Confirmed scope: train (28,130) + val (6,019) = 34,149, exactly matching the
devkit's own reported total sample count for v1.0-trainval -- generating for ALL
850 scenes (this script's default, unfiltered) covers precisely 100% of what's on
disk, nothing excluded, nothing wasted.

Usage:
    # Cheap correctness/timing test first -- 2 scenes only, before the real run:
    PYTHONNOUSERSITE=1 python scripts/generate_depth_gt_full.py --limit-scenes 2

    # Full run, all 850 scenes (only after the test run above looks correct):
    PYTHONNOUSERSITE=1 python scripts/generate_depth_gt_full.py
"""
import argparse
import os
import time

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.nuscenes_mini import (
    build_all_scene_frames, get_sample_sensor_paths, CAM_NAMES, LIDAR_NAME,
)

NUSCENES_ROOT = "/media/user/1TSSD/data/nuscenes"  # real, full v1.0-trainval data
DEPTH_GT_DIR = "/media/user/1TSSD/min/depth_gt"     # OUR OWN separate output folder --
                                                     # not written into Ruby's data/
IMG_W, IMG_H = 1600, 900  # nuScenes native resolution, same as the mini-scale script


def _sensor2global(calib_dict, pose_dict):
    sensor2ego = np.eye(4)
    sensor2ego[:3, :3] = Quaternion(calib_dict["rotation"]).rotation_matrix
    sensor2ego[:3, 3] = np.array(calib_dict["translation"])
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict["rotation"]).rotation_matrix
    ego2global[:3, 3] = np.array(pose_dict["translation"])
    return ego2global @ sensor2ego


def project_lidar_to_camera(nusc, sample_token, cam_name):
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"][LIDAR_NAME])
    lidar_calib = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    lidar2global = _sensor2global(lidar_calib, lidar_pose)

    cam_sd = nusc.get("sample_data", sample["data"][cam_name])
    cam_calib = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    cam2img = np.eye(4)
    cam2img[:3, :3] = np.asarray(cam_calib["camera_intrinsic"])
    img2cam = np.linalg.inv(cam2img)
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(cam_calib["rotation"]).rotation_matrix
    cam2ego[:3, 3] = np.asarray(cam_calib["translation"])
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(cam_pose["rotation"]).rotation_matrix
    ego2global[:3, 3] = np.asarray(cam_pose["translation"])
    img2global = ego2global @ cam2ego @ img2cam

    lidar2img = np.linalg.inv(img2global) @ lidar2global

    lidar_path = os.path.join(nusc.dataroot, lidar_sd["filename"])
    pc = LidarPointCloud.from_file(lidar_path)
    pts_lidar = pc.points[:3, :].T
    pts_hom = np.concatenate([pts_lidar, np.ones((pts_lidar.shape[0], 1))], axis=1)
    pts_img_scaled = (lidar2img @ pts_hom.T).T
    depth = pts_img_scaled[:, 2]
    valid = depth > 0.1
    u = pts_img_scaled[valid, 0] / depth[valid]
    v = pts_img_scaled[valid, 1] / depth[valid]
    d = depth[valid]
    in_bounds = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    u, v, d = u[in_bounds], v[in_bounds], d[in_bounds]
    return np.stack([u, v, d], axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-scenes", type=int, default=None,
                        help="process only the first N scenes -- for a cheap "
                             "correctness/timing test before the full 850-scene run")
    args = parser.parse_args()

    print(f"Loading NuScenes v1.0-trainval from {NUSCENES_ROOT} ...")
    nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_ROOT, verbose=True)

    scene_frames = build_all_scene_frames(nusc)
    scene_names = list(scene_frames.keys())
    print(f"\nConfirmed {len(scene_names)} scenes available (expect 850 for the full run)")

    if args.limit_scenes is not None:
        scene_names = scene_names[:args.limit_scenes]
        print(f"--limit-scenes set: processing only {len(scene_names)} scene(s) "
              f"for this test run")

    os.makedirs(DEPTH_GT_DIR, exist_ok=True)

    total, skipped = 0, 0
    t_start = time.time()
    for si, scene_name in enumerate(scene_names):
        tokens = scene_frames[scene_name]
        for tok in tokens:
            sensor_paths = get_sample_sensor_paths(nusc, tok)
            for cam in CAM_NAMES:
                img_basename = os.path.basename(sensor_paths[cam])
                out_path = os.path.join(DEPTH_GT_DIR, img_basename + ".bin")
                if os.path.exists(out_path):
                    skipped += 1
                    continue
                point_depth = project_lidar_to_camera(nusc, tok, cam)
                point_depth.tofile(out_path)
                total += 1
        elapsed = time.time() - t_start
        print(f"[{si+1}/{len(scene_names)}] {scene_name}: done "
              f"(elapsed {elapsed:.1f}s, {total} written, {skipped} skipped)")

    elapsed = time.time() - t_start
    print(f"\nWrote {total} depth files, skipped {skipped} already-existing, "
          f"to {DEPTH_GT_DIR}")
    print(f"Total time: {elapsed:.1f}s across {len(scene_names)} scene(s) "
          f"({elapsed/max(len(scene_names),1):.2f}s/scene average)")
    if args.limit_scenes is not None:
        est_full = elapsed / max(len(scene_names), 1) * 850
        print(f"\nEstimated time for the FULL 850-scene run, extrapolated from this "
              f"test: ~{est_full/60:.1f} minutes ({est_full/3600:.2f} hours)")


if __name__ == "__main__":
    main()
