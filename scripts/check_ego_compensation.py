"""
scripts/check_ego_compensation.py

Isolates whether the ego-motion compensation itself is geometrically correct,
independent of any learning. Tests three variants against do-nothing:
  - ego_only: known rigid transform (pose_prev, pose_curr), zero learned residual
  - ego_only_swapped: same but with pose order swapped -- tests for a prev/curr mixup
  - ego_only_noclamp: same as ego_only but with the pc_range clamp DISABLED -- tests
    whether the clamp (designed for small learned deltas) is corrupting the much
    larger rigid ego shift by clamping a border-band of Gaussians onto the box wall

Run from repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/check_ego_compensation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import src.training.stage_b_engine as ts1  # noqa: E402

from mmengine import Config  # noqa: E402
from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402
from loss import OPENOCC_LOSS  # noqa: E402

import json  # noqa: E402


def main():
    nusc = load_nuscenes(os.path.join(ts1.REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(ts1.REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        full_frame_index = json.load(f)

    heldout_clip_datasets = ts1.build_heldout_clip_datasets(nusc, full_frame_index)
    print(f"Held-out: {ts1.HELDOUT_SCENES}, "
          f"{sum(len(d) for d in heldout_clip_datasets)} clips total")

    cfg = Config.fromfile(
        os.path.join(ts1.GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = ts1.build_stage_a(cfg)
    encoder = ts1.CurrentFrameEncoder(segmentor)
    pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout = ts1.build_temporal_module()
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    donothing = ts1.evaluate_heldout(
        segmentor, encoder, pool, hypernet, deform_mu, deform_r,
        feature_dropout, z_dropout, cfg, loss_func, heldout_clip_datasets, mode="donothing"
    )
    ego_only = ts1.evaluate_heldout(
        segmentor, encoder, pool, hypernet, deform_mu, deform_r,
        feature_dropout, z_dropout, cfg, loss_func, heldout_clip_datasets, mode="ego_only"
    )
    print(f"do-nothing: adjusted mIoU = {donothing:.3f}")
    print(f"ego_only (WITH opacity-zeroing fix): adjusted mIoU = {ego_only:.3f}")
    print(f"delta ego_only: {ego_only - donothing:+.3f}")
    return  # skip the intentionally-crashing noclamp diagnostic below, not needed now

    import torch as _torch

    def evaluate_ego_only_swapped():
        modules = (pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout)
        for m in modules:
            m.eval()
        miou = ts1.MeanIoU(list(range(1, 17)), 17, ts1.CLASS_NAMES, True, 17, filter_minmax=False)
        miou.reset()
        with _torch.no_grad():
            for clip_dataset in heldout_clip_datasets:
                for clip_idx in range(len(clip_dataset)):
                    frame0_dict, frame1_dict = clip_dataset[clip_idx]
                    cuda0 = ts1.to_cuda(ts1.to_batch_of_one(frame0_dict))
                    cuda1 = ts1.to_cuda(ts1.to_batch_of_one(frame1_dict))
                    g0, g0_dict = ts1.get_real_g0(segmentor, cuda0)
                    pose_prev = cuda0["metas"]["lidar2global"][0]
                    pose_curr = cuda1["metas"]["lidar2global"][0]
                    relative_transform = ts1.compute_relative_transform(pose_curr, pose_prev)
                    zero_mu = _torch.zeros(g0.means.shape[-2], 3, device=g0.means.device)
                    zero_r = _torch.zeros(g0.means.shape[-2], 4, device=g0.means.device)
                    zero_r[:, 0] = 1.0
                    g1 = ts1.apply_ego_compensated_update_rule(
                        g0, zero_mu, zero_r, relative_transform, pc_range=ts1.PC_RANGE
                    )
                    _, _, head_out = ts1.splat_and_loss(segmentor, g1, type(g0_dict), cuda1, cfg, loss_func)
                    gt_occ = head_out["sampled_label"][0]
                    mask = head_out["occ_mask"].flatten(1)[0].bool()
                    pred = head_out["pred_occ"][-1][0].argmax(0)
                    miou._after_step(pred, gt_occ, mask)
        for m in modules:
            m.train()
        return ts1.adjusted_miou(miou, ts1.CLASS_NAMES)

    ego_only_swapped = evaluate_ego_only_swapped()

    def evaluate_ego_only_noclamp():
        modules = (pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout)
        for m in modules:
            m.eval()
        miou = ts1.MeanIoU(list(range(1, 17)), 17, ts1.CLASS_NAMES, True, 17, filter_minmax=False)
        miou.reset()
        clamped_frac_list = []
        with _torch.no_grad():
            for clip_dataset in heldout_clip_datasets:
                for clip_idx in range(len(clip_dataset)):
                    frame0_dict, frame1_dict = clip_dataset[clip_idx]
                    cuda0 = ts1.to_cuda(ts1.to_batch_of_one(frame0_dict))
                    cuda1 = ts1.to_cuda(ts1.to_batch_of_one(frame1_dict))
                    g0, g0_dict = ts1.get_real_g0(segmentor, cuda0)
                    pose_prev = cuda0["metas"]["lidar2global"][0]
                    pose_curr = cuda1["metas"]["lidar2global"][0]
                    relative_transform = ts1.compute_relative_transform(pose_prev, pose_curr)
                    zero_mu = _torch.zeros(g0.means.shape[-2], 3, device=g0.means.device)
                    zero_r = _torch.zeros(g0.means.shape[-2], 4, device=g0.means.device)
                    zero_r[:, 0] = 1.0
                    g1 = ts1.apply_ego_compensated_update_rule(
                        g0, zero_mu, zero_r, relative_transform, pc_range=None
                    )
                    lo = _torch.tensor(ts1.PC_RANGE[:3], device=g1.means.device)
                    hi = _torch.tensor(ts1.PC_RANGE[3:], device=g1.means.device)
                    out_of_range = ((g1.means < lo) | (g1.means > hi)).any(dim=-1).float().mean()
                    clamped_frac_list.append(out_of_range.item())
                    _, _, head_out = ts1.splat_and_loss(segmentor, g1, type(g0_dict), cuda1, cfg, loss_func)
                    gt_occ = head_out["sampled_label"][0]
                    mask = head_out["occ_mask"].flatten(1)[0].bool()
                    pred = head_out["pred_occ"][-1][0].argmax(0)
                    miou._after_step(pred, gt_occ, mask)
        for m in modules:
            m.train()
        mean_clamped_frac = sum(clamped_frac_list) / len(clamped_frac_list)
        return ts1.adjusted_miou(miou, ts1.CLASS_NAMES), mean_clamped_frac

    ego_only_noclamp, mean_out_of_range_frac = evaluate_ego_only_noclamp()

    print(f"\ndo-nothing:                                 adjusted mIoU = {donothing:.3f}")
    print(f"ego_only (pose_prev, pose_curr):            adjusted mIoU = {ego_only:.3f}")
    print(f"ego_only_swapped (pose_curr, pose_prev):    adjusted mIoU = {ego_only_swapped:.3f}")
    print(f"ego_only_noclamp (pc_range clamp disabled): adjusted mIoU = {ego_only_noclamp:.3f}")
    print(f"mean fraction of Gaussians out-of-range (no clamp): {mean_out_of_range_frac:.3f}")
    print(f"delta ego_only:         {ego_only - donothing:+.3f}")
    print(f"delta ego_only_swapped: {ego_only_swapped - donothing:+.3f}")
    print(f"delta ego_only_noclamp: {ego_only_noclamp - donothing:+.3f}")


if __name__ == "__main__":
    main()
