"""
Loads Occ3D-nuScenes voxel GT for a given sample token. Reuses loader logic from the prior
3DGS static project (bring that code over rather than reimplementing).

See docs/IMPLEMENTATION_ROADMAP.md Phase 1, step 2, and docs/dataset_compute_addendum.md
Section 2 for the known mini_train index-39 coverage gap this must handle gracefully
(return None / has_gt=False rather than raising, so the frame-index builder in
build_frame_index() can filter cleanly).

TODO(Phase 1):
    - load voxel label array for a sample token from data/occ3d_gts/scene-XXXX/<token>
    - return None if not found (do not raise) so callers can tag has_gt=False
    - apply camera visibility mask if configs/dataset_mini_occ3d.yaml: use_camera_visibility_mask
"""
