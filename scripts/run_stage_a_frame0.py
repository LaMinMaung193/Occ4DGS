"""
Phase 2: run GaussianFormer3D's Stage A (BEVSegmentorLiDAR3D) on one scene's frame 0,
using our own Occ3D-mini dataset instead of their SurroundOcc pkl pipeline.

Skips custom_collate_fn_temporal deliberately for this first run -- batch_size=1
avoids the variable-LiDAR-point-count stacking question entirely (see
EXPERIMENT_LOG.md Phase 2). Revisit collate_fn once multi-sample batching is needed.

Phase 5 repo-layout cleanup: build_pipeline/to_batch_of_one/REPO_ROOT/GF3D_ROOT moved
to src/datasets/gf3d_pipeline.py (they were being imported by 7 other scripts --
genuinely shared library code, not specific to this file's own Phase 2 demo). This
file now only keeps its own Phase 2 demo logic.
"""
import os
import sys

# Bootstrap: every entry-point script under scripts/ needs this one line before it can
# `import src...` at all, since this repo isn't pip-installed -- matches the existing
# pattern in scripts/train_stage1.py etc. Must happen BEFORE the src.* import below.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import torch  # noqa: E402

from src.datasets.gf3d_pipeline import REPO_ROOT, GF3D_ROOT, build_pipeline, to_batch_of_one  # noqa: E402

from mmengine import Config  # noqa: E402
from mmseg.models import build_segmentor  # noqa: E402

import model  # noqa: E402 -- triggers all @MODELS.register_module()/@SEGMENTORS.register_module() decorators

from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402
from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402


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
    batch["metas"]["occ_xyz"] = batch["metas"]["occ_xyz"].cuda()
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
