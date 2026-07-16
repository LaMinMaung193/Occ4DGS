"""
scripts/generate_depth_gt.py

Phase 2: generate BEVDepth-style ground-truth depth files from our own v1.0-mini LiDAR
sweeps, replacing the SharePoint-downloaded depth_gt/ files GaussianFormer3D's
LoadMultiViewDepthFromFiles expects (dataset/transform_3d.py).

Confirmed format (Phase 2 investigation, EXPERIMENT_LOG.md):
    <dataroot>/depth_gt/<image_basename>.bin
    float32 binary, np.fromfile(...).reshape(-1, 3) -> (N, 3) rows of
    (u_pixel, v_pixel, depth_meters).

Option A (decided): files are written directly into data/nuscenes_mini/depth_gt/, i.e.
onto the external drive via the existing symlink, as a sibling of samples/ -- exactly
where LoadMultiViewDepthFromFiles looks, unmodified. Additive only; does not touch any
existing nuScenes files.

Assumption (explicit -- revisit if depth quality looks sparse/poor in sanity checks):
uses only the keyframe's own single LIDAR_TOP sweep for projection, not the aggregated
10-sweep cloud used elsewhere in the pipeline. This matches standard BEVDepth practice:
aggregating sweeps for a depth map tied to one image timestamp introduces motion-blur
ghosting from the ego vehicle and other moving objects.
"""
import os
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.nuscenes_mini import (
    load_nuscenes, build_all_scene_frames, CAM_NAMES, LIDAR_NAME, get_sample_sensor_paths,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUSCENES_ROOT = os.path.join(REPO_ROOT, "data", "nuscenes_mini")

IMG_W, IMG_H = 1600, 900  # nuScenes native resolution, confirmed Phase 1 spot-check



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

    # Match GaussianFormer3D's dataset/utils.py get_img2global EXACTLY:
    #   cam2img holds K (not K^-1) -- img2cam = inv(cam2img) = K^-1
    #   img2global = ego2global @ cam2ego @ img2cam
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
    nusc = load_nuscenes(NUSCENES_ROOT)
    scene_frames = build_all_scene_frames(nusc)
    depth_gt_dir = os.path.join(NUSCENES_ROOT, "depth_gt")
    os.makedirs(depth_gt_dir, exist_ok=True)

    total, skipped = 0, 0
    for scene_name, tokens in scene_frames.items():
        for tok in tokens:
            sensor_paths = get_sample_sensor_paths(nusc, tok)
            for cam in CAM_NAMES:
                img_basename = os.path.basename(sensor_paths[cam])
                out_path = os.path.join(depth_gt_dir, img_basename + ".bin")
                if os.path.exists(out_path):
                    skipped += 1
                    continue
                point_depth = project_lidar_to_camera(nusc, tok, cam)
                point_depth.tofile(out_path)
                total += 1
        print(f"{scene_name}: done")

    print(f"\nWrote {total} depth files, skipped {skipped} already-existing, to {depth_gt_dir}")


if __name__ == "__main__":
    main()