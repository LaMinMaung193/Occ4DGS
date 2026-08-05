"""
src/datasets/gf3d_pipeline.py

Extracted from scripts/run_stage_a_frame0.py during the Phase 5 repo-layout cleanup
(EXPERIMENT_LOG.md) -- this was quietly acting as a shared library (7 other scripts
imported build_pipeline/to_batch_of_one/GF3D_ROOT directly from it), which only worked
by accident of it living in scripts/ and being importable via sys.path side effects.
Moved here so it's a real, intentional src/ module.

Behavior preserved EXACTLY from the original -- including the os.chdir(GF3D_ROOT) side
effect at import time, a known quirk (several consumers already work around it by
saving/restoring cwd around this import). Not fixed here deliberately: this is a pure
reorganization, not a behavior change, so any breakage is attributable to the move
alone. Revisit the chdir side effect as its own separate, focused fix later.
"""
import os
import sys
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # compute FIRST, before any chdir
GF3D_ROOT = os.path.expanduser("~/Documents/min/GaussianFormer3D")
sys.path.insert(0, GF3D_ROOT)
sys.path.insert(0, REPO_ROOT)

os.chdir(GF3D_ROOT)  # safe now -- REPO_ROOT already captured as an absolute path above


def build_pipeline():
    """
    Pipeline transforms reused directly from GaussianFormer3D, matching
    config/_base_/surroundocc_pcd_dfa3d.py's train_pipeline structure but with
    LoadOccupancySurroundOcc swapped for LoadOccupancyOcc3d.
    """
    from dataset.transform_3d import (
        LoadPointFromFileLiDAR, LoadPointsFromMultiSweepsLiDAR,
        LoadMultiViewImageFromFiles, LoadOccupancyOcc3d,
        LoadMultiViewDepthFromFiles, NormalizeMultiviewImage,
        PadMultiViewImage, NuScenesAdaptor,
    )
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    occ4dgs_data_root = os.path.join(REPO_ROOT, "data", "occ3d_gts")
    return [
        LoadPointFromFileLiDAR(coord_type="LIDAR", load_dim=5, use_dim=5),
        LoadPointsFromMultiSweepsLiDAR(sweeps_num=10, load_dim=5, use_dim=5,
                                        pad_empty_sweeps=True, remove_close=True),
        LoadMultiViewImageFromFiles(to_float32=True),
        LoadOccupancyOcc3d(occ_path=occ4dgs_data_root, semantic=True, use_ego=True,
                            use_occ3d_mask=True, pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                            use_lidar=True, use_mask_training=False),
        LoadMultiViewDepthFromFiles(is_to_depth_map=True, map_size=None),
        NormalizeMultiviewImage(**img_norm_cfg),
        PadMultiViewImage(size_divisor=32),
        NuScenesAdaptor(use_ego=False, num_cams=6),
    ]


def to_batch_of_one(sample_dict):
    """Wrap a single Occ4DGSDataset __getitem__ output into batch-of-1 tensors,
    bypassing custom_collate_fn_temporal for this first single-sample run."""
    imgs = torch.stack([torch.from_numpy(im).permute(2, 0, 1).float() for im in sample_dict["img"]])
    imgs = imgs.unsqueeze(0)  # (1, N, C, H, W)

    metas = {
        "projection_mat": torch.from_numpy(sample_dict["projection_mat"]).unsqueeze(0).float(),
        "image_wh": torch.from_numpy(sample_dict["image_wh"]).unsqueeze(0).float(),
        "occ_xyz": torch.from_numpy(sample_dict["occ_xyz"]).unsqueeze(0).float(),
        "occ_label": torch.from_numpy(sample_dict["occ_label"]).unsqueeze(0).long(),
        "occ_cam_mask": torch.from_numpy(sample_dict["occ_cam_mask"]).unsqueeze(0).bool(),
    }
    if "ego2global" in sample_dict:
        metas["ego2global"] = torch.from_numpy(sample_dict["ego2global"]).unsqueeze(0).float()
    if "lidar2global" in sample_dict:
        metas["lidar2global"] = torch.from_numpy(sample_dict["lidar2global"]).unsqueeze(0).float()

    points = [torch.from_numpy(sample_dict["points"].tensor.numpy()
                                if hasattr(sample_dict["points"], "tensor")
                                else sample_dict["points"]).float()]

    dpt = None
    if "dpt" in sample_dict:
        dpt = torch.stack([torch.from_numpy(d).float() for d in sample_dict["dpt"]]).unsqueeze(0).unsqueeze(2)

    return dict(imgs=imgs, metas=metas, points=points, dpt=dpt)
