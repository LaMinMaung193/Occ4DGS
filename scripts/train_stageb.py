"""
scripts/train_stageb.py

Real Stage B training loop, living in Occ4DGS (not GaussianFormer3D) --
correctly reaches OUT to GaussianFormer3D via sys.path, matching the same
dependency direction src/datasets/gf3d_pipeline.py already established for
Stage A, rather than the reverse (a mistake in earlier draft scripts, caught
and fixed here).

Trains on the 700-scene train split ONLY (an earlier draft manifest
incorrectly combined train+val into one pool with no held-out set at all --
caught and fixed in build_stageb_manifest.py before any real training run).
Evaluates on the held-out val split after every epoch -- loss only for now;
real occupancy metrics (mIoU) are a natural next addition once this loop is
confirmed working.

WARMUP_EPOCHS=2 (not the 5 validated at 30 epochs): with NUM_EPOCHS=9, a
5-epoch warmup would consume over half of training. 2/9 keeps a similar
proportion (~22%) to what was validated (5/30, ~17%) -- reasoned, not
independently re-validated at this specific ratio.

Run from Occ4DGS repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/train_stageb.py
"""
import json
import math
import os
import pickle
import sys
import time

os.environ["WANDB_MODE"] = "disabled"

import torch
import wandb
# MeanIoU._after_epoch() (GaussianFormer3D's own code, not ours) calls
# wandb.log() unconditionally, assuming wandb.init() was already called (as
# train.py does). We don't use wandb -- our own loss_history.json + print-
# based logging already covers this. Setting WANDB_MODE=disabled alone is
# NOT enough -- wandb's own "preinit" check (confirmed via the real error)
# requires wandb.init() to have actually been called at least once,
# regardless of the mode env var. mode="disabled" here means this is a
# genuine no-op: no account, no network activity, no login required.
wandb.init(mode="disabled")

GF3D_ROOT = os.path.expanduser("~/Documents/min/GaussianFormer3D")
sys.path.insert(0, GF3D_ROOT)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(GF3D_ROOT)

from mmengine import Config
from mmseg.models import build_segmentor
import model  # noqa: F401
from loss import OPENOCC_LOSS
from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal

from src.models.stage_b_temporal.current_frame_encoder import CurrentFrameEncoder
from src.models.stage_b_temporal.gf3d_faithful_deform import GF3DFaithfulDeform
from src.models.stage_b_temporal.buffer import GaussianState
from src.models.stage_b_temporal.deform_heads import transform_anchor_for_projection
from src.datasets.stageb_dataset import StageBTrainingDataset

CONFIG_PATH = os.path.join(GF3D_ROOT, "config/nuscenes_surroundocc_gs25600_full.py")
CHECKPOINT = os.path.join(GF3D_ROOT, "out/nuscenes_surroundocc_gs25600_full/epoch_3.pth")

STAGEB_DIR = "/media/user/1TSSD/min/stageb_training"
TRAIN_PAIRS_PKL = os.path.join(STAGEB_DIR, "nuscenes_infos_gf3d_stageb_pairs_train.pkl")
VAL_PAIRS_PKL = os.path.join(STAGEB_DIR, "nuscenes_infos_gf3d_stageb_pairs_val.pkl")
TRAIN_MANIFEST = os.path.join(STAGEB_DIR, "stageb_manifest_train.json")
VAL_MANIFEST = os.path.join(STAGEB_DIR, "stageb_manifest_val.json")
G0_CACHE_DIR = "/media/user/1TSSD/min/g0_cache"
OUT_DIR = os.path.join(STAGEB_DIR, "checkpoints")

N_G = 25600
EMBED_DIMS = 128
NUM_BLOCKS = 4

LR = 1e-4
MIN_LR = 1e-5  # cosine decay floor, added when extending training past 20
                # epochs -- train_loss was still improving smoothly with no
                # plateau, so this isn't a "stuck, need help" fix; it's a
                # principled, risk-reducing addition (matches Stage A's own
                # real cosine-decay convention) targeting the val_mIoU noise
                # seen late in the constant-LR run (epoch 19 dipping before
                # 20 recovered) -- decay only ever makes steps smaller, never
                # larger, so it cannot make anything worse than constant LR.
WARMUP_EPOCHS = 2
NUM_EPOCHS = 40  # extended from 20 -- epoch 20 crossed the do-nothing
              # baseline (14.46 vs 13.52 mIoU) with train_loss still
              # improving smoothly, no plateau yet; extending further to
              # find where it actually levels off
CHECKPOINT_EVERY = 3

ANCHOR_ENCODER_CFG = dict(
    type="SparseGaussian3DEncoder", embed_dims=128, include_opa=True,
    semantics=True, semantic_dim=17,
)
NORM_CFG = dict(type="LN", normalized_shape=128)
FFN_CFG = dict(
    type="AsymmetricFFN", in_channels=256, embed_dims=128,
    feedforward_channels=512, num_fcs=2, ffn_drop=0.1,
    act_cfg=dict(type="ReLU", inplace=True), pre_norm=dict(type="LN"),
)
DEFORMABLE_MODEL_CFG = dict(
    type="DeformableFeatureAggregation3D", embed_dims=128,
    kps_generator=dict(
        type="SparseGaussian3DKeyPointsGenerator3D", embed_dims=128,
        phi_activation="sigmoid", xyz_coordinate="cartesian", num_learnable_pts=2,
        fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], [0, 0.45, 0],
                   [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]],
        pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        scale_range=[0.01, 1.8],
    ),
    d_bound=[2.0, 58, 0.5], im2col_step=32, use_visibility=False,
    use_sampling_offsets=True, num_pts_per_keypoint=2, value_projection=False,
    num_cams=6, num_groups=4, num_levels=4, residual_mode="cat",
    use_camera_embed=True, use_deformable_func=True, attn_drop=0.15,
)


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


def run_one_sample(sample, encoder, deform, segmentor, loss_func, cfg, optimizer=None,
                    return_head_out=False):
    data_next = custom_collate_fn_temporal([sample["data_next"]])
    data_next = move_dict_to_cuda(data_next)
    pose_prev = sample["pose_prev"].cuda()
    pose_curr = sample["pose_curr"].cuda()

    imgs = data_next.pop("img")
    dpt = data_next.pop("dpt") if "dpt" in data_next else None
    with torch.no_grad():
        ms_img_feats, dpt_dist, out_dpt_multiscale = encoder.encode(imgs, dpt, data_next)

    g0_means = sample["g0_means"].cuda()
    g0_rotations = sample["g0_rotations"].cuda()
    g0_scales = sample["g0_scales"].cuda()
    g0_opacities = sample["g0_opacities"].cuda()
    g0_semantics = sample["g0_semantics"].cuda()
    with torch.no_grad():
        mu_proj, r_proj = transform_anchor_for_projection(
            g0_means, g0_rotations, pose_prev, pose_curr
        )
        # SAFETY CLAMP -- see baseline_do_nothing.py for full explanation.
        # Confirmed via a real crash on the full val set: some scenes with
        # large ego-motion genuinely push Gaussians outside pc_range.
        eps = 0.01
        pc_range_min = torch.tensor([-50.0 + eps, -50.0 + eps, -5.0 + eps], device=mu_proj.device)
        pc_range_max = torch.tensor([50.0 - eps, 50.0 - eps, 3.0 - eps], device=mu_proj.device)
        mu_proj = torch.clamp(mu_proj, min=pc_range_min, max=pc_range_max)
    g_prev = GaussianState(
        means=mu_proj, rotations=r_proj, scales=g0_scales,
        opacities=g0_opacities, semantics=g0_semantics,
    )

    grad_ctx = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with grad_ctx:
        if optimizer is not None:
            optimizer.zero_grad()
        g_1 = deform(g_prev, ms_img_feats, out_dpt_multiscale, data_next)

        from model.encoder.gaussian_encoder.utils import GaussianPrediction
        gaussian_pred = GaussianPrediction(
            means=g_1.means.unsqueeze(0), scales=g_1.scales.unsqueeze(0),
            rotations=g_1.rotations.unsqueeze(0), opacities=g_1.opacities.unsqueeze(0),
            semantics=g_1.semantics.unsqueeze(0),
        )
        representation = [{"gaussian": gaussian_pred}]
        head_out = segmentor.head(representation=representation, metas=data_next)

        loss_input = {"metas": data_next}
        for k, v in cfg.loss_input_convertion.items():
            loss_input[k] = head_out[v]
        loss, loss_dict = loss_func(loss_input)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    if return_head_out:
        return loss.item(), head_out
    return loss.item()


def run_evaluation(val_dataset, encoder, deform, segmentor, loss_func, cfg, miou_metric):
    """Also computes real mIoU/iou2 -- reusing GF3D's own real MeanIoU class
    and exact call pattern from train.py (confirmed against real source), the
    same mechanism that produced Stage A's own reported 23.61 mIoU. Loss and
    mIoU measure genuinely different things: loss is a continuous,
    differentiable training signal; mIoU is the real, human-interpretable,
    non-differentiable evaluation metric."""
    deform.eval()
    losses = []
    miou_metric.reset()

    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        loss_val, head_out = run_one_sample(
            sample, encoder, deform, segmentor, loss_func, cfg, optimizer=None,
            return_head_out=True,
        )
        losses.append(loss_val)

        pred = head_out["pred_occ"][-1][0]
        pred_occ = pred.argmax(0)
        gt_occ = head_out["sampled_label"][0]
        if "occ3d_mask_camera" in head_out:
            miou_metric._after_step(pred_occ, gt_occ, head_out["occ3d_mask_camera"])
        else:
            miou_metric._after_step(pred_occ, gt_occ)

    miou, iou2 = miou_metric._after_epoch()
    return sum(losses) / len(losses), miou, iou2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = Config.fromfile(CONFIG_PATH)

    print("Loading Stage A checkpoint (frozen)...")
    segmentor = build_segmentor(cfg.model)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    segmentor.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    segmentor = segmentor.cuda().eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)
    encoder = CurrentFrameEncoder(segmentor)

    print("Building real OccupancyLoss...")
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    print("Building Stage B TRAIN dataset (train split only)...")
    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = TRAIN_PAIRS_PKL
    train_underlying = OPENOCC_DATASET.build(ds_config)
    with open(TRAIN_PAIRS_PKL, "rb") as f:
        train_raw_infos = pickle.load(f)["infos"]
    train_dataset = StageBTrainingDataset(
        train_underlying, TRAIN_MANIFEST, G0_CACHE_DIR, train_raw_infos
    )

    print("Building Stage B VAL dataset (held-out, NEVER trained on)...")
    ds_config_val = dict(cfg.val_dataset_config)
    ds_config_val["imageset"] = VAL_PAIRS_PKL
    val_underlying = OPENOCC_DATASET.build(ds_config_val)
    with open(VAL_PAIRS_PKL, "rb") as f:
        val_raw_infos = pickle.load(f)["infos"]
    val_dataset = StageBTrainingDataset(
        val_underlying, VAL_MANIFEST, G0_CACHE_DIR, val_raw_infos
    )

    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
        True, 17, filter_minmax=False,
    )

    print(f"\nBuilding GF3DFaithfulDeform (L={NUM_BLOCKS})...")
    deform = GF3DFaithfulDeform(
        num_blocks=NUM_BLOCKS, embed_dims=EMBED_DIMS, num_anchor=N_G,
        anchor_encoder_cfg=ANCHOR_ENCODER_CFG,
        deformable_model_cfg=DEFORMABLE_MODEL_CFG,
        norm_cfg=NORM_CFG, ffn_cfg=FFN_CFG,
    ).cuda()
    optimizer = torch.optim.AdamW(deform.parameters(), lr=LR)

    # RESUME LOGIC: check for an existing checkpoint before starting fresh.
    # Without this, any crash-and-restart (accidental Ctrl+C, OOM, power
    # blip) would silently discard all progress and start over from random
    # init -- only safe to auto-restart once this exists.
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_iou2": []}
    existing_checkpoints = sorted(
        [f for f in os.listdir(OUT_DIR) if f.startswith("epoch_") and f.endswith(".pth")],
        key=lambda f: int(f[len("epoch_"):-len(".pth")]),
    ) if os.path.isdir(OUT_DIR) else []

    if existing_checkpoints:
        latest = existing_checkpoints[-1]
        latest_path = os.path.join(OUT_DIR, latest)
        print(f"\nFound existing checkpoint: {latest_path}")
        ckpt = torch.load(latest_path, map_location="cuda")
        deform.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"]
        print(f"Resuming from epoch {start_epoch + 1} (already completed: {start_epoch})")
    else:
        print("\nNo existing checkpoint found -- starting fresh from epoch 1.")

    print(f"\n{'='*70}")
    print(f"Training: {len(train_dataset)} train scenes, {len(val_dataset)} held-out val scenes")
    print(f"{NUM_EPOCHS} epochs total, warmup={WARMUP_EPOCHS} epochs, lr={LR}")
    print(f"Output: {OUT_DIR}")
    print(f"{'='*70}\n")

    t_total_start = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        if epoch < WARMUP_EPOCHS:
            current_lr = LR * (epoch + 1) / WARMUP_EPOCHS
        else:
            # Cosine decay LR -> MIN_LR over the post-warmup range. Note:
            # epochs 2-19 already ran at constant LR (added only when
            # extending past 20) -- this does not "redo" those epochs, it
            # only changes the formula used for epochs from here forward.
            # At epoch 20 specifically, this lands partway down the curve
            # (~5.7e-5, not a fresh 1e-4) -- a real, visible step down in
            # the log, not a discontinuity to worry about.
            progress = (epoch - WARMUP_EPOCHS) / max(1, (NUM_EPOCHS - WARMUP_EPOCHS - 1))
            progress = min(progress, 1.0)
            current_lr = MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        deform.train()
        order = torch.randperm(len(train_dataset)).tolist()
        epoch_losses = []
        t_epoch_start = time.time()

        for i, idx in enumerate(order):
            sample = train_dataset[idx]
            loss_val = run_one_sample(
                sample, encoder, deform, segmentor, loss_func, cfg, optimizer,
            )
            epoch_losses.append(loss_val)
            if (i + 1) % 100 == 0:
                print(f"    epoch {epoch+1} | {i+1}/{len(order)} scenes | "
                      f"running_avg_train_loss={sum(epoch_losses)/len(epoch_losses):.4f}")

        train_avg = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - t_epoch_start

        print(f"  Running held-out evaluation ({len(val_dataset)} val scenes)...")
        t_eval_start = time.time()
        val_avg, val_miou, val_iou2 = run_evaluation(
            val_dataset, encoder, deform, segmentor, loss_func, cfg, miou_metric
        )
        eval_time = time.time() - t_eval_start

        history["train_loss"].append(train_avg)
        history["val_loss"].append(val_avg)
        history["val_miou"].append(float(val_miou))
        history["val_iou2"].append(float(val_iou2))
        elapsed = time.time() - t_total_start

        print(f"  epoch {epoch+1:3d}/{NUM_EPOCHS} | lr={current_lr:.2e} | "
              f"train_loss={train_avg:8.4f} | val_loss={val_avg:8.4f} | "
              f"val_mIoU={val_miou:.4f} | val_iou2={val_iou2:.4f} | "
              f"epoch_time={epoch_time/60:.1f}min | eval_time={eval_time/60:.1f}min | "
              f"total_elapsed={elapsed/60:.1f}min")

        history_path = os.path.join(OUT_DIR, "loss_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == NUM_EPOCHS:
            ckpt_path = os.path.join(OUT_DIR, f"epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": deform.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, ckpt_path)
            print(f"    saved checkpoint -> {ckpt_path}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Epoch 1:  train={history['train_loss'][0]:.4f}  val_loss={history['val_loss'][0]:.4f}  "
          f"val_mIoU={history['val_miou'][0]:.4f}")
    print(f"  Epoch {NUM_EPOCHS}: train={history['train_loss'][-1]:.4f}  val_loss={history['val_loss'][-1]:.4f}  "
          f"val_mIoU={history['val_miou'][-1]:.4f}")
    best_miou_epoch = history["val_miou"].index(max(history["val_miou"])) + 1
    print(f"  Best val_mIoU: {max(history['val_miou']):.4f} (epoch {best_miou_epoch})")
    print(f"  Total time: {(time.time()-t_total_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
