"""
Loads Occ3D-nuScenes voxel GT for a given sample token. Reuses loader logic from the prior
3DGS static project (bring that code over rather than reimplementing).

Confirmed file structure (Phase 0, 2026-07-14, EXPERIMENT_LOG.md):
    data/occ3d_gts/<scene-name>/<sample-token>/labels.npz contains THREE separate arrays,
    not one combined label + mask:
        - semantics:    (200, 200, 16) uint8, values 0-17 (0-16 semantic classes, 17 = free)
        - mask_lidar:   (200, 200, 16) uint8, binary — voxels with LiDAR returns
        - mask_camera:  (200, 200, 16) uint8, binary — voxels visible to any camera

    mask_camera matches configs/dataset_mini_occ3d.yaml's use_camera_visibility_mask flag.
    mask_lidar was not originally accounted for in configs — it's a candidate input for
    L_lidar (Phase 6, docs/design_doc_v2.md Section 4) since it tells you which voxels
    actually had LiDAR observations, rather than inferring that from geometry alone.

See docs/IMPLEMENTATION_ROADMAP.md Phase 1, step 2, and docs/dataset_compute_addendum.md
Section 2 for the known mini_train index-39 coverage gap this must handle gracefully
(return None / has_gt=False rather than raising, so the frame-index builder in
build_frame_index() can filter cleanly).

TODO(Phase 1):
    - load labels.npz for a sample token from data/occ3d_gts/scene-XXXX/<token>/labels.npz
    - return a dict with keys {semantics, mask_lidar, mask_camera}, or None if the file
      doesn't exist (do not raise) so callers can tag has_gt=False
    - apply mask_camera when use_camera_visibility_mask is set (configs/dataset_mini_occ3d.yaml)
    - surface mask_lidar separately so Phase 6's L_lidar implementation can use it directly
      rather than rediscovering this file structure from scratch
"""
