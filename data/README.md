# data/

This folder is gitignored except this README. On the actual training machine, symlink:

    ln -s /media/user/Transcend/nuScenes/v1.0-mini      data/nuscenes_mini
    ln -s /media/user/Transcend/data/occ3d/gts          data/occ3d_gts

Do NOT point anything here at /media/user/Transcend/data123 (full trainval blobs, blobs 04/05
incomplete) or /media/user/Transcend/data/nuscenes (partial CAM_BACK+LIDAR_TOP only) — see
docs/dataset_compute_addendum.md Section 1 for why those are excluded from this project.
