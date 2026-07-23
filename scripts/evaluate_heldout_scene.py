"""
scripts/evaluate_heldout_scene.py

Tests generalization: loads the exact Stage A checkpoint (stage_a_best.pth) and the
exact trained Stage B temporal module (stage1_warmup_temporal.pth) -- both trained ONLY
on scene-0061 -- and runs the same do-nothing-vs-trained mIoU comparison on a scene
NEITHER was ever trained on. No training happens in this script at all; retraining on
the held-out scene would just be a second in-sample fit, not a generalization test.

If the trained-vs-do-nothing gap holds up here (even if smaller than the in-sample
scene-0061 numbers), that's real evidence Stage B learned something about motion
prediction generally, not scene-0061 specifically. If the gap collapses to ~0 or
reverses, the prior result was likely overfitting to that one scene's specific patterns.

Run from repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/evaluate_heldout_scene.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from mmengine import Config  # noqa: E402

from train_stage1 import (  # noqa: E402 -- reusing exact helper functions, not duplicating logic
    build_stage_a,
    build_temporal_module,
    get_real_g0,
    deform_one_step,
    splat_and_loss,
    to_cuda,
    CLASS_NAMES,
    GF3D_ROOT,
    REPO_ROOT,
)
from mmseg.models import build_segmentor  # noqa: E402,F401 -- imported for cfg.model build inside build_stage_a
import model  # noqa: E402,F401
from loss import OPENOCC_LOSS  # noqa: E402
from misc.metric_util import MeanIoU  # noqa: E402

from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402
from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402
from src.datasets.occ4dgs_clip_dataset import Occ4DGSClipDataset  # noqa: E402

import json  # noqa: E402

from run_stage_a_frame0 import build_pipeline, to_batch_of_one  # noqa: E402


HELDOUT_SCENE = "scene-0103"  # confirmed distinct from SCENES=["scene-0061"] used to
                                # train both Stage A and Stage B -- true held-out test


def main():
    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        full_frame_index = json.load(f)
    frame_index = {HELDOUT_SCENE: full_frame_index[HELDOUT_SCENE]}

    base_dataset = Occ4DGSDataset(
        nusc, frame_index,
        os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
        os.path.join(REPO_ROOT, "data", "occ3d_gts"),
        pipeline=build_pipeline(),
    )
    clip_dataset = Occ4DGSClipDataset(base_dataset, unroll_window=2)
    print(f"HELD-OUT scene {HELDOUT_SCENE}: {len(clip_dataset)} clips "
          f"(never used to train Stage A or Stage B)")

    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_stage_a(cfg)  # loads stage_a_best.pth, frozen+eval, per train_stage1.py

    pool, hypernet, deform_mu, deform_r = build_temporal_module()
    temporal_ckpt_path = os.path.join(
        REPO_ROOT, "experiments", "stage_b_temporal_checkpoints", "stage1_warmup_temporal.pth"
    )
    temporal_ckpt = torch.load(temporal_ckpt_path, map_location="cuda")
    pool.load_state_dict(temporal_ckpt["pool"])
    hypernet.load_state_dict(temporal_ckpt["hypernet"])
    deform_mu.load_state_dict(temporal_ckpt["deform_mu"])
    deform_r.load_state_dict(temporal_ckpt["deform_r"])
    for m in (pool, hypernet, deform_mu, deform_r):
        m.eval()
    print(f"Loaded trained temporal module (trained on {temporal_ckpt['trained_on_scenes']}, "
          f"{temporal_ckpt['n_epochs']} epochs): {temporal_ckpt_path}")

    from train_stage1 import CurrentFrameEncoder  # noqa: E402 -- avoid duplicate import at top
    encoder = CurrentFrameEncoder(segmentor)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    miou_trained = MeanIoU(list(range(1, 17)), 17, CLASS_NAMES, True, 17, filter_minmax=False)
    miou_donothing = MeanIoU(list(range(1, 17)), 17, CLASS_NAMES, True, 17, filter_minmax=False)
    miou_trained.reset()
    miou_donothing.reset()

    with torch.no_grad():
        for clip_idx in range(len(clip_dataset)):
            frame0_dict, frame1_dict = clip_dataset[clip_idx]
            cuda0 = to_cuda(to_batch_of_one(frame0_dict))
            cuda1 = to_cuda(to_batch_of_one(frame1_dict))

            g0, g0_dict = get_real_g0(segmentor, cuda0)

            g1_trained = deform_one_step(g0, encoder, pool, hypernet,
                                          deform_mu, deform_r, cuda1)
            _, _, head_out_trained = splat_and_loss(
                segmentor, g1_trained, type(g0_dict), cuda1, cfg, loss_func
            )
            _, _, head_out_donothing = splat_and_loss(
                segmentor, g0, type(g0_dict), cuda1, cfg, loss_func
            )

            gt_occ = head_out_trained["sampled_label"][0]
            mask = head_out_trained["occ_mask"].flatten(1)[0].bool()
            pred_trained = head_out_trained["pred_occ"][-1][0].argmax(0)
            pred_donothing = head_out_donothing["pred_occ"][-1][0].argmax(0)

            miou_trained._after_step(pred_trained, gt_occ, mask)
            miou_donothing._after_step(pred_donothing, gt_occ, mask)

    miou_t, iou2_t = miou_trained._after_epoch()
    miou_d, iou2_d = miou_donothing._after_epoch()
    print(f"\n=== HELD-OUT ({HELDOUT_SCENE}) RESULTS ===")
    print(f"Trained:    mIoU={miou_t}, iou2={iou2_t}")
    print(f"Do-nothing: mIoU={miou_d}, iou2={iou2_d}")
    print(f"Delta:      mIoU={miou_t - miou_d:+.4f}, iou2={iou2_t - iou2_d:+.4f}")
    print("\nCompare against scene-0061 in-sample: Trained=16.252/28.808, "
          "Do-nothing=14.898/23.943, Delta=+1.354/+4.865")


if __name__ == "__main__":
    main()