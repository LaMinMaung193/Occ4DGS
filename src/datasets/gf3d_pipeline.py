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
import cv2
import mmcv
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # compute FIRST, before any chdir
GF3D_ROOT = os.path.expanduser("~/Documents/min/GaussianFormer3D")
sys.path.insert(0, GF3D_ROOT)
sys.path.insert(0, REPO_ROOT)

os.chdir(GF3D_ROOT)  # safe now -- REPO_ROOT already captured as an absolute path above


class PadRawImagesForVGGT:
    """
    NEW transform, written for this project -- NOT one of GaussianFormer3D's real
    dataset/transform_3d.py classes (none of those support a configurable output key,
    and none of the existing ones fit VGGT's different input contract).

    VGGT_DEFORMABLE_DESIGN.md's implementation-blocking decisions (both confirmed
    against real source before writing this):
      - PAD, not resize, images before VGGT (size_divisor=14, VGGT's patch size) --
        keeps every real pixel at its true, calibration-matching coordinate, exactly
        matching how PadMultiViewImage(size_divisor=32) already pads for the existing
        ResNet/FPN backbone. Reuses mmcv.impad_to_multiple verbatim -- the same
        padding primitive PadMultiViewImage itself uses internally -- just applied to
        a SEPARATE copy of the raw images, under new keys, so this never collides
        with or mutates results['img'] (which NormalizeMultiviewImage/
        PadMultiViewImage(32) continue to process, unchanged, for the ResNet path).
      - VGGT's real expected input (confirmed against vggt/models/aggregator.py AND
        vggt/utils/load_fn.py, the official preprocessing utility, not assumed): raw
        [0,1]-scale RGB, NOT ImageNet mean/std normalized -- Aggregator.forward does
        that normalization internally. Feeding it pre-normalized pixels (like
        results['img'] after NormalizeMultiviewImage) would double-normalize and
        silently corrupt every downstream feature -- exactly why this transform MUST
        run BEFORE NormalizeMultiviewImage, while results['img'] is still raw BGR
        (confirmed via LoadMultiViewImageFromFiles's own to_rgb=False metadata at
        that point -- the BGR->RGB flip only happens later, inside
        NormalizeMultiviewImage itself).

    DOWNSAMPLE FIX (found on real training, not anticipated by the design doc): the
    native padded resolution (910x1610, 65x115=7475 patches/frame) is ~5.5x more
    patches per frame than VGGT-1B's own tested/default scale (img_size=518,
    37x37=1369 patches -- confirmed against vggt/models/aggregator.py's real default,
    and against the README's own "VGGT typically reconstructs a scene in less than 1
    second" claim). Self-attention cost scales quadratically with sequence length, so
    feeding 12 frames (6 cams x 2 timesteps, Section 2's "both frames together"
    decision) at native resolution gave a ~90,000-token global-attention sequence --
    confirmed via real training on the lab machine: 100% GPU utilization (genuine
    compute, not a hang), ~22 minutes per optimizer step, projecting to ~9 days for
    one Gate 2 (n=3) run.

    Fix: downsample AFTER padding, to (VGGT_TARGET_H, VGGT_TARGET_W) -- landing at
    28x50=1400 patches/frame, close to VGGT's own native 1369. This does NOT reopen
    the resize-vs-pad distortion concern from the original decision: that concern was
    about a slightly non-uniform resize at NATIVE resolution, awkwardly hitting an
    odd multiple-of-14 target. This is a clean, separate step on an ALREADY correctly
    padded canvas -- and critically, project_points_3d's [0,1] normalization divides
    x by width and y by height INDEPENDENTLY (not a single combined "scale"), so
    geometric correctness holds exactly regardless of whether the H and W downsample
    factors match each other (they happen to be close here -- 2.32 vs 2.30 -- but
    that's incidental, not required), as long as vggt_image_wh is set to the TRUE
    final (post-downsample) resolution, which it is, below.

    Must run in build_pipeline() AFTER LoadMultiViewImageFromFiles (needs
    results['img'] populated) and BEFORE NormalizeMultiviewImage (needs the raw,
    un-normalized, still-BGR pixel values).

    Adds:
        results['vggt_img']: list of (VGGT_TARGET_H, VGGT_TARGET_W, 3) float32
            arrays, RGB order, [0,1] scale, padded then downsampled. NOT
            ImageNet-normalized.
        results['vggt_image_wh']: (num_cams, 2) float32 array of
            [VGGT_TARGET_W, VGGT_TARGET_H] per camera -- the TRUE final resolution
            VGGT's own dense features actually span, for project_points_3d's
            normalization. Deliberately NOT the same array as results['image_wh']
            (32-padded, native resolution, for the ResNet path) -- these are two
            different resolutions for two different networks and must not be
            conflated (VGGT_DEFORMABLE_DESIGN.md's own consistency note).
    """

    def __init__(self, size_divisor=14, target_h=392, target_w=700):
        self.size_divisor = size_divisor
        # 28*14=392, 50*14=700 -- both exact multiples of size_divisor (required for
        # VGGT's patch embedding), landing at 28*50=1400 patches/frame, close to
        # VGGT-1B's own native 37*37=1369 -- see class docstring for the full
        # derivation and why H/W downsample factors don't need to match each other.
        assert target_h % size_divisor == 0 and target_w % size_divisor == 0, (
            f"target_h={target_h}/target_w={target_w} must both be exact multiples "
            f"of size_divisor={size_divisor} -- VGGT's patch embedding requires this."
        )
        self.target_h = target_h
        self.target_w = target_w

    def __call__(self, results):
        raw_bgr_views = results["img"]  # list of (H,W,3) float32, BGR, un-normalized
        vggt_views = []
        native_padded_h, native_padded_w = None, None
        for img in raw_bgr_views:
            rgb_01 = img[..., ::-1].astype(np.float32) / 255.0  # BGR->RGB, scale to [0,1]
            padded = mmcv.impad_to_multiple(rgb_01, self.size_divisor, pad_val=0.0)
            native_padded_h, native_padded_w = padded.shape[:2]  # BEFORE downsampling -- see below
            # INTER_AREA: the standard, recommended OpenCV interpolation for
            # DOWNSAMPLING specifically (better anti-aliasing than bilinear when
            # shrinking) -- this is real image content, not a discrete label map.
            downsampled = cv2.resize(padded, (self.target_w, self.target_h),
                                      interpolation=cv2.INTER_AREA)
            vggt_views.append(downsampled)
        results["vggt_img"] = vggt_views
        # BUG FIX (found via real training -- an ~84-86% fallback-embedding usage
        # rate, traced to this exact line): vggt_image_wh must reflect the NATIVE-
        # PADDED resolution (pre-downsample, e.g. 1610x910), NOT the downsampled
        # target (700x392) -- project_points_3d's [0,1] normalization divides
        # projection_mat's own pixel-coordinate output by image_wh, and
        # projection_mat ALWAYS outputs native-scale pixel coordinates (it is never
        # touched by padding or downsampling). Dividing native-scale coordinates by
        # a downsampled width/height gives values far outside [0,1] for almost
        # every real point, which is exactly what caused the near-total fallback
        # rate. grid_sample itself never receives image_wh at all -- it only needs
        # the [-1,1] grid coordinate (resolution-independent by construction) and
        # the actual tensor's own shape, which it already has -- so downsampling
        # the IMAGE TENSOR while keeping image_wh at native-padded scale is
        # correct and was the original intent, just not what got implemented.
        results["vggt_image_wh"] = np.ascontiguousarray(
            np.array([[float(native_padded_w), float(native_padded_h)]] * len(vggt_views), dtype=np.float32)
        )
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(size_divisor={self.size_divisor})"


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
        PadRawImagesForVGGT(size_divisor=14),  # MUST be here: after raw load, before Normalize mutates 'img'
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
    # VGGT_DEFORMABLE_DESIGN.md: a SEPARATE padded resolution (multiple of 14, not 32)
    # and SEPARATE, non-ImageNet-normalized pixel values from imgs/image_wh above --
    # deliberately not reusing or deriving from them (see PadRawImagesForVGGT).
    if "vggt_image_wh" in sample_dict:
        metas["vggt_image_wh"] = torch.from_numpy(sample_dict["vggt_image_wh"]).unsqueeze(0).float()

    points = [torch.from_numpy(sample_dict["points"].tensor.numpy()
                                if hasattr(sample_dict["points"], "tensor")
                                else sample_dict["points"]).float()]

    dpt = None
    if "dpt" in sample_dict:
        dpt = torch.stack([torch.from_numpy(d).float() for d in sample_dict["dpt"]]).unsqueeze(0).unsqueeze(2)

    vggt_imgs = None
    if "vggt_img" in sample_dict:
        vggt_imgs = torch.stack(
            [torch.from_numpy(im).permute(2, 0, 1).float() for im in sample_dict["vggt_img"]]
        ).unsqueeze(0)  # (1, N, C, H_pad14, W_pad14) -- own padding, distinct from imgs above

    return dict(imgs=imgs, metas=metas, points=points, dpt=dpt, vggt_imgs=vggt_imgs)
