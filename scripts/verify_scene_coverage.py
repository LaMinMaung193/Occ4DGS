"""
Phase 0, step 4: confirm all 10 v1.0-mini scene names exist as folders in data/occ3d_gts/.
"""
import os
from nuscenes.nuscenes import NuScenes

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUSCENES_ROOT = os.path.join(REPO_ROOT, "data", "nuscenes_mini")
OCC3D_GTS_ROOT = os.path.join(REPO_ROOT, "data", "occ3d_gts")


def main():
    nusc = NuScenes(version="v1.0-mini", dataroot=NUSCENES_ROOT, verbose=True)
    mini_scene_names = sorted(s["name"] for s in nusc.scene)
    print(f"Found {len(mini_scene_names)} scenes in v1.0-mini:")
    for name in mini_scene_names:
        print(" ", name)

    occ3d_scene_folders = set(os.listdir(OCC3D_GTS_ROOT))
    print(f"\nFound {len(occ3d_scene_folders)} scene folders under {OCC3D_GTS_ROOT}")

    missing = [s for s in mini_scene_names if s not in occ3d_scene_folders]
    if missing:
        print(f"\nMISSING Occ3D GT for {len(missing)} scenes:")
        for m in missing:
            print("  ", m)
        raise SystemExit(1)
    print("\nAll 10 v1.0-mini scenes have matching Occ3D GT folders. Phase 0 step 4: PASS")


if __name__ == "__main__":
    main()