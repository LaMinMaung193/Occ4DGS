"""
scripts/check_vggt_data_plumbing.py

Standalone sanity check for the new VGGT data stream (PadRawImagesForVGGT +
to_batch_of_one + Occ4DGSDataset's whitelist + stage_b_engine.to_cuda), BEFORE any
VGGT model code exists. Confirms the plumbing itself is correct on one real sample:
shapes, dtype, value range, padding-divisibility, and that vggt_imgs is genuinely
DIFFERENT data from imgs (different normalization/padding), not an accidental alias
or a silently-empty stream.

Run: PYTHONNOUSERSITE=1 python scripts/check_vggt_data_plumbing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import torch  # noqa: E402

from src.datasets.gf3d_pipeline import REPO_ROOT, build_pipeline, to_batch_of_one  # noqa: E402
from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402
from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402
from src.training.stage_b_engine import to_cuda  # noqa: E402


def main():
    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        full_frame_index = json.load(f)
    scene_name = next(iter(full_frame_index))
    frame_index = {scene_name: full_frame_index[scene_name]}

    dataset = Occ4DGSDataset(
        nusc, frame_index,
        os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
        os.path.join(REPO_ROOT, "data", "occ3d_gts"),
        pipeline=build_pipeline(),
    )
    sample_dict = dataset[0]

    print("[1/6] Occ4DGSDataset.__getitem__ produced the new keys:")
    for k in ("vggt_img", "vggt_image_wh"):
        assert k in sample_dict, f"MISSING: '{k}' not in sample_dict -- whitelist edit didn't take"
        print(f"      '{k}' present")

    batch = to_batch_of_one(sample_dict)
    assert batch["vggt_imgs"] is not None, "to_batch_of_one produced vggt_imgs=None"
    print(f"[2/6] to_batch_of_one: vggt_imgs shape {tuple(batch['vggt_imgs'].shape)}, "
          f"metas['vggt_image_wh'] shape {tuple(batch['metas']['vggt_image_wh'].shape)}")

    N_cam = batch["imgs"].shape[1]
    _, _, _, H14, W14 = batch["vggt_imgs"].shape
    assert H14 % 14 == 0 and W14 % 14 == 0, (
        f"vggt_imgs spatial dims ({H14},{W14}) not divisible by 14 -- padding is wrong"
    )
    print(f"[3/6] [PASS] vggt_imgs spatial dims ({H14},{W14}) both divisible by 14")

    _, _, _, H32, W32 = batch["imgs"].shape
    print(f"      (for reference: existing imgs padded shape is ({H32},{W32}), divisible by 32: "
          f"{H32 % 32 == 0 and W32 % 32 == 0})")

    vmin, vmax = batch["vggt_imgs"].min().item(), batch["vggt_imgs"].max().item()
    print(f"[4/6] vggt_imgs value range: [{vmin:.4f}, {vmax:.4f}] (expect within [0, 1])")
    assert -1e-6 <= vmin and vmax <= 1.0 + 1e-6, (
        f"vggt_imgs values outside [0,1] -- likely still ImageNet-normalized or not /255'd"
    )

    imgs_min, imgs_max = batch["imgs"].min().item(), batch["imgs"].max().item()
    print(f"      (for reference: existing imgs value range: [{imgs_min:.2f}, {imgs_max:.2f}] "
          f"-- should NOT look like [0,1], confirming they're genuinely different streams)")
    assert not (-1e-6 <= imgs_min and imgs_max <= 1.0 + 1e-6), (
        "existing imgs ALSO look like [0,1] -- something is wrong, these should be "
        "ImageNet-normalized (roughly [-2,2] range), not raw [0,1]"
    )
    print("[5/6] [PASS] vggt_imgs and imgs are genuinely different data (different scale/normalization)")

    cuda_batch = to_cuda(batch)
    assert cuda_batch["vggt_imgs"] is not None
    assert cuda_batch["vggt_imgs"].is_cuda
    assert cuda_batch["metas"]["vggt_image_wh"].is_cuda
    print("[6/6] [PASS] to_cuda moves vggt_imgs and metas['vggt_image_wh'] to GPU correctly")

    print(f"\nAll VGGT data plumbing checks passed. N_cam={N_cam}, "
          f"vggt_image_wh[0,0]={cuda_batch['metas']['vggt_image_wh'][0,0].tolist()}")


if __name__ == "__main__":
    main()
