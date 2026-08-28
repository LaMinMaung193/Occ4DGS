# data/

This folder is gitignored except this README. On the actual training machine, symlink:

    ln -s /media/user/Transcend/nuScenes/v1.0-mini      data/nuscenes_mini
    ln -s /media/user/Transcend/data/occ3d/gts          data/occ3d_gts

Do NOT point anything here at /media/user/Transcend/data123 (full trainval blobs, blobs 04/05
incomplete) or /media/user/Transcend/data/nuscenes (partial CAM_BACK+LIDAR_TOP only) — see
docs/dataset_compute_addendum.md Section 1 for why those are excluded from this project.

## Full-scale dataset (added — full-dataset pivot)
Full-scale nuScenes v1.0-trainval + SurroundOcc data lives on /media/user/1TSSD,
symlinked inside the SEPARATE GaussianFormer3D repo's own data/ folder (not here --
GF3D's own train.py is used directly for full-scale Stage A, not this repo's
mini-dataset pipeline). See EXPERIMENT_LOG.md for the full setup history.
The mini-dataset setup above remains valid and is kept deliberately, as a fast,
cheap sanity-check tool for new Stage B designs before committing to full-scale runs.
