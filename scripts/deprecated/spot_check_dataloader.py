"""
Phase 1 exit checklist item 3: manually verify camera/LiDAR/GT shapes for one
spot-checked, GT-valid sample from each of the 10 scenes.
"""
import os, sys
from PIL import Image
from nuscenes.utils.data_classes import LidarPointCloud

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.nuscenes_mini import load_nuscenes, build_all_scene_frames, get_sample_sensor_paths, CAM_NAMES, LIDAR_NAME
from src.datasets.occ3d_gt import load_occ3d_labels

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUSCENES_ROOT = os.path.join(REPO_ROOT, "data", "nuscenes_mini")
OCC3D_ROOT = os.path.join(REPO_ROOT, "data", "occ3d_gts")


def main():
    nusc = load_nuscenes(NUSCENES_ROOT)
    scene_frames = build_all_scene_frames(nusc)

    for scene_name, tokens in scene_frames.items():
        chosen = None
        for tok in tokens:
            gt = load_occ3d_labels(OCC3D_ROOT, scene_name, tok)
            if gt is not None:
                chosen = (tok, gt)
                break
        if chosen is None:
            print(f"{scene_name}: NO valid-GT sample found at all -- investigate")
            continue

        tok, gt = chosen
        paths = get_sample_sensor_paths(nusc, tok)
        img = Image.open(paths[CAM_NAMES[0]])
        pc = LidarPointCloud.from_file(paths[LIDAR_NAME])

        print(f"{scene_name}: token={tok[:8]}...")
        print(f"  {CAM_NAMES[0]} image size: {img.size}")
        print(f"  LiDAR points shape: {pc.points.shape}")  # (4, N): x,y,z,intensity
        print(f"  GT semantics shape: {gt['semantics'].shape}, dtype: {gt['semantics'].dtype}")


if __name__ == "__main__":
    main()