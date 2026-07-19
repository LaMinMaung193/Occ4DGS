"""
Phase 2: run GaussianFormer3D's Stage A (BEVSegmentorLiDAR3D) on one scene's frame 0,
using our own Occ3D-mini dataset instead of their SurroundOcc pkl pipeline.

Skips custom_collate_fn_temporal deliberately for this first run -- batch_size=1
avoids the variable-LiDAR-point-count stacking question entirely (see
EXPERIMENT_LOG.md Phase 2). Revisit collate_fn once multi-sample batching is needed.
"""
import os, sys, json
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # compute FIRST, before any chdir
GF3D_ROOT = os.path.expanduser("~/Documents/min/GaussianFormer3D")
sys.path.insert(0, GF3D_ROOT)
sys.path.insert(0, REPO_ROOT)

os.chdir(GF3D_ROOT)  # safe now -- REPO_ROOT already captured as an absolute path above


from mmengine import Config
from mmseg.models import build_segmentor

import model  # <-- ADD THIS: triggers all @MODELS.register_module()/@SEGMENTORS.register_module() decorators

from src.datasets.nuscenes_mini import load_nuscenes
from src.datasets.occ4dgs_dataset import Occ4DGSDataset



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
    occ4dgs_data_root = os.path.join(REPO_ROOT, "data", "occ3d_gts")   # <-- FIXED: was just "data"
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

    points = [torch.from_numpy(sample_dict["points"].tensor.numpy()
                                if hasattr(sample_dict["points"], "tensor")
                                else sample_dict["points"]).float()]

    dpt = None
    if "dpt" in sample_dict:
        dpt = torch.stack([torch.from_numpy(d).float() for d in sample_dict["dpt"]]).unsqueeze(0).unsqueeze(2)

    return dict(imgs=imgs, metas=metas, points=points, dpt=dpt)


def main():
    frame_index_path = os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")
    with open(frame_index_path) as f:
        frame_index = json.load(f)

    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    pipeline = build_pipeline()

    dataset = Occ4DGSDataset(
        nusc=nusc,
        frame_index=frame_index,
        nuscenes_root=os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
        occ3d_gts_root=os.path.join(REPO_ROOT, "data", "occ3d_gts"),
        pipeline=pipeline,
    )

    # First sample of the first scene only -- Phase 2 step 2 (one scene, frame 0).
    scene0_indices = [i for i, s in enumerate(dataset.samples) if s[0] == "scene-0061"]
    sample = dataset[scene0_indices[0]]
    print("Dataset sample keys:", list(sample.keys()))
    print("occ_label shape:", sample["occ_label"].shape)

    batch = to_batch_of_one(sample)
    print("imgs shape:", batch["imgs"].shape)
    print("projection_mat shape:", batch["metas"]["projection_mat"].shape)
    print("points[0] shape:", batch["points"][0].shape)
    if batch["dpt"] is not None:
        print("dpt shape:", batch["dpt"].shape)

    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    model = build_segmentor(cfg.model)
    model.init_weights()
    model = model.cuda()

    for k in ["imgs", "points"]:
        batch[k] = [t.cuda() for t in batch[k]] if isinstance(batch[k], list) else batch[k].cuda()
    batch["metas"]["projection_mat"] = batch["metas"]["projection_mat"].cuda()
    batch["metas"]["image_wh"] = batch["metas"]["image_wh"].cuda()
    batch["metas"]["occ_xyz"] = batch["metas"]["occ_xyz"].cuda()  # <-- ADD THIS TOO
    batch["metas"]["occ_label"] = batch["metas"]["occ_label"].cuda()
    batch["metas"]["occ_cam_mask"] = batch["metas"]["occ_cam_mask"].cuda()
    if batch["dpt"] is not None:
        batch["dpt"] = batch["dpt"].cuda()

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model(**batch)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    print("\nOutput keys:", list(out.keys()))
    print("Peak VRAM: {:.2f} GB".format(peak_vram_gb))


if __name__ == "__main__":
    main()