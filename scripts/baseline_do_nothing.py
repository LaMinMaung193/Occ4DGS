"""
scripts/baseline_do_nothing.py

"Do-nothing" baseline for Stage B: evaluates real mIoU/iou2 using G_0
(frame-transformed via transform_anchor_for_projection, same preprocessing
Stage B itself receives) directly as the prediction for frame-next -- with
NO deformation network involved at all. Gives a genuine reference point:
Stage B training is only adding real value if its own val_mIoU rises above
this baseline, not just because a number goes up in isolation.

Reuses the exact same real components as train_stageb.py (Stage A checkpoint,
CurrentFrameEncoder, GaussianHead splatting, MeanIoU) on the SAME held-out
140-scene val set, so the comparison is genuinely apples-to-apples.

Run from Occ4DGS repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/baseline_do_nothing.py
"""
import os
import pickle
import sys

GF3D_ROOT = os.path.expanduser("~/Documents/min/GaussianFormer3D")
sys.path.insert(0, GF3D_ROOT)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(GF3D_ROOT)

os.environ["WANDB_MODE"] = "disabled"

import torch
import wandb
wandb.init(mode="disabled")

from mmengine import Config
from mmseg.models import build_segmentor
import model  # noqa: F401
from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal

from src.models.stage_b_temporal.current_frame_encoder import CurrentFrameEncoder
from src.models.stage_b_temporal.buffer import GaussianState
from src.models.stage_b_temporal.deform_heads import transform_anchor_for_projection
from src.datasets.stageb_dataset import StageBTrainingDataset

CONFIG_PATH = os.path.join(GF3D_ROOT, "config/nuscenes_surroundocc_gs25600_full.py")
CHECKPOINT = os.path.join(GF3D_ROOT, "out/nuscenes_surroundocc_gs25600_full/epoch_3.pth")

STAGEB_DIR = "/media/user/1TSSD/min/stageb_training"
VAL_PAIRS_PKL = os.path.join(STAGEB_DIR, "nuscenes_infos_gf3d_stageb_pairs_val.pkl")
VAL_MANIFEST = os.path.join(STAGEB_DIR, "stageb_manifest_val.json")
G0_CACHE_DIR = "/media/user/1TSSD/min/g0_cache"


def move_dict_to_cuda(data):
    for k in list(data.keys()):
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].cuda()
        if isinstance(data[k], dict):
            for kk in data[k]:
                if isinstance(data[k][kk], torch.Tensor):
                    data[k][kk] = data[k][kk].cuda()
        if isinstance(data[k], list):
            for kk in range(len(data[k])):
                if isinstance(data[k][kk], torch.Tensor):
                    data[k][kk] = data[k][kk].cuda()
    return data


def main():
    cfg = Config.fromfile(CONFIG_PATH)

    print("Loading Stage A checkpoint (frozen)...")
    segmentor = build_segmentor(cfg.model)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    segmentor.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    segmentor = segmentor.cuda().eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)
    encoder = CurrentFrameEncoder(segmentor)

    from misc.metric_util import MeanIoU

    def extract_per_class_ious(miou_metric):
        per_class = {}
        for i, label in enumerate(miou_metric.label_str):
            if miou_metric.total_seen[i] == 0:
                iou = 1.0
            else:
                iou = (miou_metric.total_correct[i] / (
                    miou_metric.total_seen[i] + miou_metric.total_positive[i]
                    - miou_metric.total_correct[i]
                )).item()
            per_class[label] = iou * 100
        return per_class

    miou_metric = MeanIoU(
        list(range(1, 17)),
        17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
        True, 17, filter_minmax=False,
    )
    miou_metric.reset()

    print("Building Stage B VAL dataset (full 140-scene held-out set)...")
    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = VAL_PAIRS_PKL
    val_underlying = OPENOCC_DATASET.build(ds_config)
    with open(VAL_PAIRS_PKL, "rb") as f:
        val_raw_infos = pickle.load(f)["infos"]
    val_dataset = StageBTrainingDataset(
        val_underlying, VAL_MANIFEST, G0_CACHE_DIR, val_raw_infos
    )
    print(f"  {len(val_dataset)} val scenes")

    print(f"\nRunning do-nothing baseline (G_0, frame-transformed, NO deformation)...")
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            data_next = custom_collate_fn_temporal([sample["data_next"]])
            data_next = move_dict_to_cuda(data_next)
            pose_prev = sample["pose_prev"].cuda()
            pose_curr = sample["pose_curr"].cuda()

            g0_means = sample["g0_means"].cuda()
            g0_rotations = sample["g0_rotations"].cuda()
            g0_scales = sample["g0_scales"].cuda()
            g0_opacities = sample["g0_opacities"].cuda()
            g0_semantics = sample["g0_semantics"].cuda()

            mu_proj, r_proj = transform_anchor_for_projection(
                g0_means, g0_rotations, pose_prev, pose_curr
            )
            # SAFETY CLAMP: some real Gaussians genuinely drift outside the
            # valid pc_range after a real frame-transform, for scenes with
            # large enough ego-motion -- confirmed via a real crash across
            # the full 140-scene val set (not seen in earlier 1-2 scene
            # testing). This is physically real, not a bug -- the splatting
            # kernel strictly requires in-bounds points and crashes rather
            # than handling it gracefully, so we clamp defensively here.
            eps = 0.01
            pc_range_min = torch.tensor([-50.0 + eps, -50.0 + eps, -5.0 + eps], device=mu_proj.device)
            pc_range_max = torch.tensor([50.0 - eps, 50.0 - eps, 3.0 - eps], device=mu_proj.device)
            mu_proj = torch.clamp(mu_proj, min=pc_range_min, max=pc_range_max)

            from model.encoder.gaussian_encoder.utils import GaussianPrediction
            gaussian_pred = GaussianPrediction(
                means=mu_proj.unsqueeze(0), scales=g0_scales.unsqueeze(0),
                rotations=r_proj.unsqueeze(0), opacities=g0_opacities.unsqueeze(0),
                semantics=g0_semantics.unsqueeze(0),
            )
            representation = [{"gaussian": gaussian_pred}]
            head_out = segmentor.head(representation=representation, metas=data_next)

            pred = head_out["pred_occ"][-1][0]
            pred_occ = pred.argmax(0)
            gt_occ = head_out["sampled_label"][0]
            if "occ3d_mask_camera" in head_out:
                miou_metric._after_step(pred_occ, gt_occ, head_out["occ3d_mask_camera"])
            else:
                miou_metric._after_step(pred_occ, gt_occ)

            if (idx + 1) % 20 == 0:
                print(f"  {idx+1}/{len(val_dataset)} scenes processed...")

    miou, iou2 = miou_metric._after_epoch()
    per_class = extract_per_class_ious(miou_metric)
    import json
    result = {"method": "do_nothing_baseline", "mIoU": float(miou), "iou2": float(iou2), "per_class_iou": per_class}
    out_path = "/media/user/1TSSD/min/stageb_training/eval_results/do_nothing_baseline.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n{'='*60}")
    print(f"DO-NOTHING BASELINE (G_0 unchanged, {len(val_dataset)} held-out scenes)")
    print(f"  mIoU:  {float(miou):.4f}")
    print(f"  iou2:  {float(iou2):.4f}")
    print(f"  Saved -> {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
