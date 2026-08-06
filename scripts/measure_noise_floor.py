"""
scripts/measure_noise_floor.py

Step 0 (EXPERIMENT_LOG.md 2026-07-30 noise-floor finding): isolates WHERE the
do-nothing baseline's run-to-run variance actually comes from.

Findings so far (see EXPERIMENT_LOG.md for full detail):
  - NOT random-seed sensitivity (SEED=42 was already fixed across all 4 sweep runs).
  - torch.backends.cudnn.deterministic=True alone: no improvement.
  - torch.use_deterministic_algorithms(True, warn_only=True): surfaced real
    CuBLAS-matmul non-determinism (gaussian_head.py's covariance computation) plus two
    unfixable-in-this-PyTorch-version, loss-only ops (not reachable from pred_occ/mIoU).
  - CUBLAS_WORKSPACE_CONFIG=:4096:8: confirmed fixed the CuBLAS matmul issue
    specifically, but overall range barely moved -- CuBLAS was real but not dominant.
  - bisect(): splatting/head is PERFECTLY bit-identical across repeats (fully
    exonerated). Stage A's encoder alone is NOT bit-identical -- max abs diff 76.8 on
    position values that should range [-40,40]. This magnitude is far too large to be
    ordinary floating-point reduction-order noise -- more consistent with an actual bug
    (e.g. uninitialized GPU memory in a custom kernel) than benign non-determinism.
  - bisect_fine(): narrows down further -- checks raw backbone+FPN features (standard
    conv ops, pre-DFA3D) for determinism separately from the full encoder-to-G_0 path
    (which includes DFA3D's custom deformable-attention kernel across 4 refinement
    blocks), to localize whether the break is in standard ops or specifically in DFA3D.

DECISION (per user, after weighing cost/benefit of a full kernel-level fix): this
script's job is to characterize and localize the issue as precisely as is cheaply
possible in Python -- NOT to attempt an actual CUDA/kernel-level fix, which would
require different skills, unbounded time, and would distract from the actual Step 5
research work. Once localized, the practical path is: document the finding, treat the
measured noise magnitude as a known constant, and use repeated runs / distributions
(not single-point comparisons) for any future architecture claim.

Run from repo root, in the gf3d env:
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONNOUSERSITE=1 python scripts/measure_noise_floor.py [N_REPEATS]
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONNOUSERSITE=1 python scripts/measure_noise_floor.py bisect
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONNOUSERSITE=1 python scripts/measure_noise_floor.py bisect_fine
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import torch  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

from mmengine import Config  # noqa: E402

from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402

from src.training.stage_b_engine import (  # noqa: E402
    REPO_ROOT,
    GF3D_ROOT,
    build_stage_a,
    build_temporal_module,
    build_heldout_clip_datasets,
    evaluate_heldout,
)

from loss import OPENOCC_LOSS  # noqa: E402


_arg = sys.argv[1] if len(sys.argv) > 1 else None
N_REPEATS = int(_arg) if _arg is not None and _arg not in ("bisect", "bisect_fine") else 5


def main():
    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        full_frame_index = json.load(f)
    heldout_clip_datasets = build_heldout_clip_datasets(nusc, full_frame_index)

    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_stage_a(cfg)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    pool, hypernet, deform_mu, deform_r, feature_dropout, z_dropout, spawn_head = build_temporal_module()

    print(f"Running {N_REPEATS} repeated do-nothing baseline evaluations, "
          f"same process, same checkpoint, same held-out scenes, every time...\n")

    results = []
    for i in range(N_REPEATS):
        val = evaluate_heldout(
            segmentor, None, pool, hypernet, deform_mu, deform_r,
            feature_dropout, z_dropout, cfg, loss_func,
            heldout_clip_datasets, mode="donothing", spawn_head=spawn_head,
        )
        results.append(val)
        print(f"  run {i+1}/{N_REPEATS}: do-nothing adjusted mIoU = {val:.6f}")

    print(f"\nmin={min(results):.6f}  max={max(results):.6f}  "
          f"range={max(results)-min(results):.6f}  "
          f"mean={sum(results)/len(results):.6f}")
    if max(results) - min(results) < 1e-9:
        print("\n=> PERFECTLY DETERMINISTIC within this process.")
    else:
        print("\n=> NON-DETERMINISTIC even within a single process, same checkpoint, "
              "same data, no training involved.")


def _load_clip_and_segmentor():
    from src.training.stage_b_engine import to_cuda, to_batch_of_one, build_pipeline
    from src.datasets.occ4dgs_dataset import Occ4DGSDataset
    from src.datasets.occ4dgs_clip_dataset import Occ4DGSClipDataset

    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        full_frame_index = json.load(f)
    frame_index = {"scene-1094": full_frame_index["scene-1094"]}
    base = Occ4DGSDataset(
        nusc, frame_index,
        os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
        os.path.join(REPO_ROOT, "data", "occ3d_gts"),
        pipeline=build_pipeline(),
    )
    clip_dataset = Occ4DGSClipDataset(base, unroll_window=2)
    frame0_dict, frame1_dict = clip_dataset[0]
    cuda0 = to_cuda(to_batch_of_one(frame0_dict))
    cuda1 = to_cuda(to_batch_of_one(frame1_dict))

    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    segmentor = build_stage_a(cfg)
    return segmentor, cfg, cuda0, cuda1


def bisect():
    """Isolates WHICH stage introduces non-determinism: Stage A's encoder (backbone +
    DFA3D deformable attention) producing G_0, or the splatting/head step, or both.
    """
    from src.training.stage_b_engine import get_real_g0
    segmentor, cfg, cuda0, cuda1 = _load_clip_and_segmentor()

    print("\n=== BISECTION A: Stage A encoder (backbone+DFA3D) alone, 5 repeats ===")
    means_list = []
    with torch.no_grad():
        for i in range(5):
            g0, g0_dict = get_real_g0(segmentor, cuda0)
            means_list.append(g0.means.clone())
    ref = means_list[0]
    all_identical = all(torch.equal(ref, m) for m in means_list[1:])
    max_diff = max((ref - m).abs().max().item() for m in means_list[1:])
    print(f"  bit-identical across 5 repeats: {all_identical}  (max abs diff: {max_diff:.2e})")

    print("\n=== BISECTION B: splatting/head alone, FIXED Gaussians, 5 repeats ===")
    g0_fixed, g0_dict = get_real_g0(segmentor, cuda0)
    pred_list = []
    with torch.no_grad():
        for i in range(5):
            g_wrapped = [{"gaussian": g0_dict}]
            head_out = segmentor.head(representation=g_wrapped, metas=cuda1["metas"])
            pred_list.append(head_out["pred_occ"][-1].clone())
    ref_pred = pred_list[0]
    all_identical_pred = all(torch.equal(ref_pred, p) for p in pred_list[1:])
    max_diff_pred = max((ref_pred - p).abs().max().item() for p in pred_list[1:])
    print(f"  bit-identical across 5 repeats: {all_identical_pred}  (max abs diff: {max_diff_pred:.2e})")


def bisect_fine():
    """Narrows down bisect()'s result further: checks raw backbone+FPN features
    (standard conv ops, PRE-DFA3D) for determinism, separately from the full encoder
    forward pass that includes DFA3D's custom deformable-attention kernel across the
    4 iterative Gaussian-refinement blocks. Reuses CurrentFrameEncoder.encode(), which
    conveniently exposes exactly the pre-DFA3D backbone+FPN output already (no need to
    reach into segmentor.forward()'s internals).
    """
    from src.training.stage_b_engine import CurrentFrameEncoder
    segmentor, cfg, cuda0, cuda1 = _load_clip_and_segmentor()
    encoder = CurrentFrameEncoder(segmentor)

    print("\n=== BISECTION FINE: raw backbone+FPN features alone (pre-DFA3D), 5 repeats ===")
    img_feats_list = []
    dpt_feats_list = []
    with torch.no_grad():
        for i in range(5):
            ms_img_feats, _dpt_dist, out_dpt_multiscale = encoder.encode(
                cuda0["imgs"], cuda0["dpt"], cuda0["metas"]
            )
            img_feats_list.append([f.clone() for f in ms_img_feats])
            dpt_feats_list.append([f.clone() for f in out_dpt_multiscale])

    ref_img = img_feats_list[0]
    img_identical = all(
        all(torch.equal(ref_img[lvl], run[lvl]) for lvl in range(len(ref_img)))
        for run in img_feats_list[1:]
    )
    img_max_diff = max(
        max((ref_img[lvl] - run[lvl]).abs().max().item() for lvl in range(len(ref_img)))
        for run in img_feats_list[1:]
    )
    print(f"  ms_img_feats (camera backbone+FPN) bit-identical: {img_identical}  "
          f"(max abs diff: {img_max_diff:.2e})")

    ref_dpt = dpt_feats_list[0]
    dpt_identical = all(
        all(torch.equal(ref_dpt[lvl], run[lvl]) for lvl in range(len(ref_dpt)))
        for run in dpt_feats_list[1:]
    )
    dpt_max_diff = max(
        max((ref_dpt[lvl] - run[lvl]).abs().max().item() for lvl in range(len(ref_dpt)))
        for run in dpt_feats_list[1:]
    )
    print(f"  out_dpt_multiscale (depth head) bit-identical: {dpt_identical}  "
          f"(max abs diff: {dpt_max_diff:.2e})")

    if img_identical and dpt_identical:
        print("\n=> Standard backbone+FPN+depth-head ops are CLEAN. The break is "
              "specifically inside the iterative Gaussian-refinement blocks "
              "(DFA3D's custom deformable-attention kernel, or the sparse-conv "
              "self-encoding step) -- confirms a custom-kernel source, not standard "
              "PyTorch ops.")
    else:
        print("\n=> Non-determinism already present in standard backbone/FPN/depth-head "
              "ops, BEFORE DFA3D -- a different, earlier source than expected.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bisect":
        bisect()
    elif len(sys.argv) > 1 and sys.argv[1] == "bisect_fine":
        bisect_fine()
    else:
        main()
