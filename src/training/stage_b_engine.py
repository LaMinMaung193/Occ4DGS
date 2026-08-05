"""
src/training/stage_b_engine.py

Extracted from scripts/train_stage1.py during the Phase 5 repo-layout cleanup
(EXPERIMENT_LOG.md) -- this file was quietly acting as a shared library (4 other
scripts imported functions/constants directly from it, either via
`from train_stage1 import (...)` or `import train_stage1 as ts1`), on top of being the
main training entrypoint. That dual role, combined with an UNRELATED empty stub that
happened to share the exact same name (src/training/train_stage1.py, now deleted),
was a real latent bug risk -- resolved by making this the one real, intentional home
for all the reusable Stage B engine code, with scripts/train_stage1.py now a thin
entrypoint that imports from here.

Every name a consumer previously accessed via `ts1.<name>` (either import style) is
preserved here so `import src.training.stage_b_engine as ts1` is a drop-in replacement
for the old `import train_stage1 as ts1` pattern.
"""
import os

import torch
import torch.nn as nn

# Must come BEFORE `import model`/`from misc.metric_util import ...` below -- those are
# GaussianFormer3D's own top-level modules (not pip packages), only importable once
# GF3D_ROOT is on sys.path, which this import does as a side effect. Original bug: this
# ordering was inverted during extraction, causing ModuleNotFoundError: No module named
# 'model' on the very first real run -- caught and fixed via the smoke test, exactly
# what that test was for.
from src.datasets.gf3d_pipeline import REPO_ROOT, GF3D_ROOT, build_pipeline, to_batch_of_one  # noqa: F401

from mmseg.models import build_segmentor
import model  # noqa: F401 -- triggers @MODELS.register_module()/@SEGMENTORS.register_module() decorators
from misc.metric_util import MeanIoU  # noqa: F401 -- re-exported, used by consumers via ts1.MeanIoU
from src.datasets.occ4dgs_dataset import Occ4DGSDataset
from src.datasets.occ4dgs_clip_dataset import Occ4DGSClipDataset
from src.models.stage_b_temporal import (
    GaussianState,
    ReferenceBuffer,  # noqa: F401 -- re-exported, used by scripts/train_stage1.py's main()
    MotionHyperNet,  # noqa: F401 -- re-exported, kept available for comparison/rollback
    ConvHyperNet,
    query_motion_grid,  # noqa: F401 -- re-exported, kept available for comparison/rollback
    query_motion_grid_pe_coordinate,
    DeformHeadMu,
    DeformHeadR,
    apply_update_rule,
    compute_relative_transform,
    apply_ego_compensated_update_rule,
    compute_spawn_candidate_positions,
    SpawnHead,
)
from src.models.stage_b_temporal.current_frame_encoder import CurrentFrameEncoder  # noqa: F401
from src.models.stage_b_temporal.pool_features import PoolFeatures


PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
CLASS_NAMES = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation',
]
HELDOUT_SCENES = ["scene-1094", "scene-1100"]

DROPOUT_P = 0.2

USE_EGO_COMPENSATION = False

USE_SPAWN_HEAD = True and USE_EGO_COMPENSATION
SPAWN_GRID_FEAT_DIM = 24  # Step 3: query_motion_grid_pe_coordinate doubles z's dim
                          # (sin+cos sample per level), 3 levels * 2 * 4 channels = 24
SPAWN_POOLED_DIM = 128
SPAWN_MAX_OFFSET = 2.0


def to_cuda(batch):
    out = {"imgs": batch["imgs"].cuda(), "points": [t.cuda() for t in batch["points"]]}
    out["metas"] = {k: v.cuda() for k, v in batch["metas"].items()}
    out["dpt"] = batch["dpt"].cuda() if batch["dpt"] is not None else None
    return out


def build_stage_a(cfg):
    segmentor = build_segmentor(cfg.model)
    checkpoint_path = os.path.join(
        REPO_ROOT, "experiments", "stage_a_checkpoints", "stage_a_best.pth"
    )
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        segmentor.load_state_dict(state_dict, strict=True)
        print(f"Loaded real trained Stage A checkpoint: {checkpoint_path}")
    else:
        segmentor.init_weights()
        print("WARNING: no trained Stage A checkpoint found, falling back to init_weights().")
    segmentor = segmentor.cuda()
    segmentor.eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)
    return segmentor


def build_temporal_module():
    pool = PoolFeatures(img_channels=128, dpt_channels=112, num_levels=4, in_dim=128).cuda()
    hypernet = ConvHyperNet(in_dim=128, grid_feat_dim=4, resolutions=(8, 16, 32),
                             seed_res=4, seed_channels=64).cuda()
    deform_mu = DeformHeadMu(in_dim=3 * 2 * 4, hidden_dim=128).cuda()
    deform_r = DeformHeadR(in_dim=3 * 2 * 4, hidden_dim=128, max_angle_rad=0.3).cuda()
    feature_dropout = nn.Dropout(p=DROPOUT_P).cuda()
    z_dropout = nn.Dropout(p=DROPOUT_P).cuda()
    spawn_head = None
    if USE_SPAWN_HEAD:
        spawn_head = SpawnHead(
            grid_feat_dim=SPAWN_GRID_FEAT_DIM, pooled_dim=SPAWN_POOLED_DIM,
            hidden_dim=128, semantic_dim=17, max_offset=SPAWN_MAX_OFFSET,
        ).cuda()
    return pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout, spawn_head


def get_real_g0(segmentor, cuda0):
    with torch.no_grad():
        representation0 = segmentor(
            imgs=cuda0["imgs"].clone(),
            metas=cuda0["metas"],
            points=cuda0["points"],
            dpt=cuda0["dpt"].clone() if cuda0["dpt"] is not None else None,
            rep_only=True,
        )
    g0_dict = representation0[-1]["gaussian"]
    g0 = GaussianState(
        means=g0_dict.means, rotations=g0_dict.rotations,
        scales=g0_dict.scales, opacities=g0_dict.opacities, semantics=g0_dict.semantics,
    )
    return g0, g0_dict


def deform_one_step(g_prev, encoder, pool, hypernet, deform_mu, deform_r,
                     feature_dropout, z_dropout, cuda_prev, cuda_curr,
                     spawn_head=None, no_grad_encoder=True, return_deltas=False):
    ctx = torch.no_grad() if no_grad_encoder else torch.enable_grad()
    with ctx:
        ms_img_feats_prev, _dpt_dist_prev, out_dpt_multiscale_prev = encoder.encode(
            cuda_prev["imgs"], cuda_prev["dpt"], cuda_prev["metas"]
        )
        ms_img_feats_curr, _dpt_dist_curr, out_dpt_multiscale_curr = encoder.encode(
            cuda_curr["imgs"], cuda_curr["dpt"], cuda_curr["metas"]
        )
    pooled_prev = pool(ms_img_feats_prev, out_dpt_multiscale_prev)
    pooled_curr = pool(ms_img_feats_curr, out_dpt_multiscale_curr)
    pooled_delta = feature_dropout(pooled_curr - pooled_prev)

    grids = hypernet(pooled_delta)
    means_flat = g_prev.means.squeeze(0) if g_prev.means.dim() == 3 else g_prev.means
    z = z_dropout(query_motion_grid_pe_coordinate(means_flat, grids, PC_RANGE))
    delta_mu = deform_mu(z)
    delta_r = deform_r(z)

    if not USE_EGO_COMPENSATION:
        g_t = apply_update_rule(g_prev, delta_mu, delta_r, pc_range=PC_RANGE)
        if return_deltas:
            return g_t, delta_mu, delta_r
        return g_t

    pose_prev = cuda_prev["metas"]["lidar2global"][0]
    pose_curr = cuda_curr["metas"]["lidar2global"][0]
    relative_transform = compute_relative_transform(pose_prev, pose_curr)

    spawn_offset = spawn_opacity = spawn_semantics = None
    if spawn_head is not None:
        wrapped_base, _out_of_range = compute_spawn_candidate_positions(
            means_flat, delta_mu, relative_transform, PC_RANGE
        )
        z_candidate = query_motion_grid_pe_coordinate(wrapped_base, grids, PC_RANGE)
        spawn_offset, spawn_opacity, spawn_semantics = spawn_head(z_candidate, pooled_delta)

    g_t = apply_ego_compensated_update_rule(
        g_prev, delta_mu, delta_r, relative_transform, pc_range=PC_RANGE,
        spawn_offset=spawn_offset, spawn_opacity=spawn_opacity, spawn_semantics=spawn_semantics,
    )
    if return_deltas:
        return g_t, delta_mu, delta_r
    return g_t


def splat_and_loss(segmentor, g_state, g_dict_type, cuda_frame, cfg, loss_func):
    g_wrapped = [{"gaussian": g_dict_type(
        means=g_state.means, rotations=g_state.rotations, scales=g_state.scales,
        opacities=g_state.opacities, semantics=g_state.semantics,
    )}]
    head_out = segmentor.head(representation=g_wrapped, metas=cuda_frame["metas"])
    loss_input = {}
    for k, v in cfg.loss_input_convertion.items():
        loss_input.update({k: head_out[v]})
    loss, loss_dict = loss_func(loss_input)
    return loss, loss_dict, head_out


def adjusted_miou(miou_metric, class_names):
    total_seen = miou_metric.total_seen[:-1]
    total_correct = miou_metric.total_correct[:-1]
    total_positive = miou_metric.total_positive[:-1]
    ious = []
    for i in range(len(class_names)):
        if total_seen[i].item() == 0:
            continue
        union = (total_seen[i] + total_positive[i] - total_correct[i]).item()
        ious.append((total_correct[i].item() / union) if union > 0 else 0.0)
    return (sum(ious) / len(ious) * 100) if ious else float("nan")


def build_heldout_clip_datasets(nusc, full_frame_index):
    datasets = []
    for scene_name in HELDOUT_SCENES:
        frame_index = {scene_name: full_frame_index[scene_name]}
        base = Occ4DGSDataset(
            nusc, frame_index,
            os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
            os.path.join(REPO_ROOT, "data", "occ3d_gts"),
            pipeline=build_pipeline(),
        )
        datasets.append(Occ4DGSClipDataset(base, unroll_window=2))
    return datasets


def evaluate_heldout(segmentor, encoder, pool, hypernet, deform_mu, deform_r,
                      feature_dropout, z_dropout, cfg, loss_func,
                      heldout_clip_datasets, mode, spawn_head=None):
    assert mode in ("trained", "donothing", "ego_only")
    modules = [pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout]
    if spawn_head is not None:
        modules.append(spawn_head)
    for m in modules:
        m.eval()
    miou = MeanIoU(list(range(1, 17)), 17, CLASS_NAMES, True, 17, filter_minmax=False)
    miou.reset()
    with torch.no_grad():
        for clip_dataset in heldout_clip_datasets:
            for clip_idx in range(len(clip_dataset)):
                frame0_dict, frame1_dict = clip_dataset[clip_idx]
                cuda0 = to_cuda(to_batch_of_one(frame0_dict))
                cuda1 = to_cuda(to_batch_of_one(frame1_dict))
                g0, g0_dict = get_real_g0(segmentor, cuda0)
                if mode == "trained":
                    g1 = deform_one_step(g0, encoder, pool, hypernet, deform_mu, deform_r,
                                          feature_dropout, z_dropout, cuda0, cuda1,
                                          spawn_head=spawn_head)
                elif mode == "ego_only":
                    pose_prev = cuda0["metas"]["lidar2global"][0]
                    pose_curr = cuda1["metas"]["lidar2global"][0]
                    relative_transform = compute_relative_transform(pose_prev, pose_curr)
                    zero_mu = torch.zeros(g0.means.shape[-2], 3, device=g0.means.device)
                    zero_r = torch.zeros(g0.means.shape[-2], 4, device=g0.means.device)
                    zero_r[:, 0] = 1.0
                    g1 = apply_ego_compensated_update_rule(
                        g0, zero_mu, zero_r, relative_transform, pc_range=PC_RANGE
                    )
                else:
                    g1 = g0
                _, _, head_out = splat_and_loss(segmentor, g1, type(g0_dict), cuda1, cfg, loss_func)
                gt_occ = head_out["sampled_label"][0]
                mask = head_out["occ_mask"].flatten(1)[0].bool()
                pred = head_out["pred_occ"][-1][0].argmax(0)
                miou._after_step(pred, gt_occ, mask)
    for m in modules:
        m.train()
    return adjusted_miou(miou, CLASS_NAMES)
