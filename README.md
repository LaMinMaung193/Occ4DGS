# Occ4DGS: Dynamic 4D Gaussian Splatting for Occupancy Prediction in Autonomous Driving

Feedforward temporal deformation of GaussianFormer3D primitives for dynamic 3D semantic
occupancy prediction. Full-scale training on nuScenes v1.0-trainval (850 scenes) with
SurroundOcc GT; nuScenes v1.0-mini (10 scenes) with Occ3D-nuScenes GT kept as a fast,
cheap iteration/sanity-check tool for new Stage B designs before committing to
full-scale runs.

CCU Autonomous Driving Perception Lab, advised by Prof. Rachael (Jui-Chiu) Chiang.

See `docs/IMPLEMENTATION_ROADMAP.md` for the phase-by-phase plan, `EXPERIMENT_LOG.md`
for the full running research log, `docs/design_doc_v2.md` for the original overall
project architecture/rationale, and `docs/STAGE_B_GF3D_FAITHFUL_DESIGN.md` for Stage
B's current, official architecture (professor-approved), including the full method,
math, and professor Q&A.

## Status (current)

**Stage A:** trained on the full 700-scene nuScenes v1.0-trainval + SurroundOcc set,
`N_g=25,600` (GaussianFormer3D's own real full-scale config, not the mini-dataset
`N_g=6,400`). Best checkpoint: `epoch_3.pth`, `mIoU=23.61` (GF3D's own real metric,
evaluated on a 70-scene held-out subset) — stopped deliberately at 3 of a planned 6
epochs, given the time budget; ~87% of GaussianFormer3D's own reported 27.1 mIoU,
reached in a fraction of their 24 epochs. `G_0` (Stage A's output) cached for all 850
scenes (`/media/user/1TSSD/min/g0_cache/`), so no future Stage B experiment needs to
re-run Stage A. Full derivation: `EXPERIMENT_LOG.md`, entries 2026-08-21 through
2026-08-28.

**Stage B:** architecture finalized and professor-approved as **GF3D-faithful** —
built by directly reusing GaussianFormer3D's own real modules
(`DeformableFeatureAggregation3D`, the anchor/`Q` construction, the per-block
`spconv→norm→deformable→ffn→norm→refine` sequence) for the `t>0` deformation step,
applied to the reference-buffer Gaussian instead of a freshly-initialized one — rather
than a custom-built mechanism. Full architecture, method, math, and the professor's
review Q&A: `docs/STAGE_B_GF3D_FAITHFUL_DESIGN.md`. **Implementation has not yet
started** on this branch.

**Prior architectures, concluded:** two earlier Stage B designs (a custom
`HyperNet`/grid-query mechanism, and a later VGGT-1B + deformable-attention design)
were built, extensively debugged, and gated — both showed the same underlying
signature (real in-sample training signal, but held-out performance peaking early and
never recovering), consistent across five independently-tried architectures at the
mini-dataset's ~8-10 scene scale. This cross-architecture pattern (not a single
architecture's flaw) directly motivated the move to full-scale data. Full arc:
`EXPERIMENT_LOG.md`, entries 2026-07-27 through 2026-08-19.

## Git branch structure

- `gf3d-faithful-stageb` — **active**, official report architecture, all new
  implementation happens here.
- `main` — Motion HyperNet (unchanged for now); will be replaced with GF3D-faithful
  content once implemented and confirmed working.
- `stage-a-full-dataset-gs25600` — full-dataset Stage A infrastructure
  (`gf3d-faithful-stageb` branches from this, not `main`).
- `archive/motion-hypernet-backup` — the original custom Stage B design, kept as a
  personal backup/Plan B, not part of the official report.
- `archive/vggt-deformable-attention` — the VGGT+deformable-attention design,
  concluded and discarded per the professor's direction; full history preserved.
- `archive/option-d-direct-projection` — an earlier, simpler direct-projection design,
  concluded before VGGT was tried; full history preserved.

## Architecture note: reuse, not reimplementation (now applies to both stages)

**Stage A:** the original plan assumed building `src/models/stage_a_gaussianformer3d/`
from scratch. In practice, GaussianFormer3D's own `BEVSegmentorLiDAR3D` class is
directly reusable — we write a config and a thin dataset adapter that feeds our data
into their real pipeline, rather than porting their architecture ourselves. For the
full-scale dataset, GaussianFormer3D's own `train.py`/`tools/make_gf3d_infos.py` are
used directly (in the separate `GaussianFormer3D` repo), not this repo's mini-dataset
adapter.

**Stage B (as of the GF3D-faithful pivot):** this same philosophy now extends to
Stage B. Rather than the earlier custom `HyperNet`/grid-query mechanism (kept on
`archive/motion-hypernet-backup`) or the VGGT-based design (discarded), the current
design reuses GaussianFormer3D's own real deformation modules directly, applied to
the reference-buffer Gaussian across frames. See
`docs/STAGE_B_GF3D_FAITHFUL_DESIGN.md` for the complete architecture, including
which pieces are reused verbatim vs. newly written (the reference buffer,
`CurrentFrameEncoder`, and our own `DeformHeadMu`/`DeformHeadR` heads all carry over
from the earlier designs and remain unchanged).

## Datasets

**Full-scale (current, active):** nuScenes v1.0-trainval (850 scenes) + SurroundOcc
annotations, on an external SSD (`/media/user/1TSSD/`), symlinked inside the separate
`GaussianFormer3D` repo's own `data/` folder — not tracked in this repo. Setup process
(including two real double-nesting/write-corruption issues found and fixed) fully
documented in `EXPERIMENT_LOG.md`, entries 2026-08-21 onward.

**Mini (kept deliberately, not deprecated):** nuScenes v1.0-mini (10 scenes) +
Occ3D-nuScenes GT, symlinked per `data/README.md`. Retained specifically as a fast,
cheap sanity-check tool for new Stage B designs — confirm correctness/wiring here
before committing to an expensive full-scale run, following the same gated,
cost-controlled philosophy used throughout this project's architecture search (see
`EXPERIMENT_LOG.md`'s Gate 1-4 protocol, adopted 2026-07-30).

## Repository structure (current, gf3d-faithful-stageb)

configs/
    occ4dgs_mini_occ3d_gs6400.py  -- mini-dataset config, kept as dev/iteration tool
data/
    README.md  -- mini-dataset symlink setup + full-scale pointer
docs/
    design_doc_v2.md  -- original overall project architecture/rationale
    design_doc_v1_superseded.md  -- earlier version, kept for history
    STAGE_B_GF3D_FAITHFUL_DESIGN.md  -- Stage B's current architecture (to be added)
    IMPLEMENTATION_ROADMAP.md
    dataset_compute_addendum.md
    EXPERIMENT_LOG_TEMPLATE.md
src/datasets/
    gf3d_pipeline.py  -- shared GF3D_ROOT-dependent import/build helpers
    nuscenes_mini.py  -- mini-dataset scene/frame indexing
    occ3d_gt.py  -- Occ3D GT loader
    occ4dgs_dataset.py  -- mini-dataset adapter feeding GaussianFormer3D's real
        pipeline; this is Stage A in practice
    occ4dgs_clip_dataset.py  -- wraps Occ4DGSDataset into temporally-adjacent frame
        pairs; Motion-HyperNet-era Stage B, not yet updated for GF3D-faithful
src/eval/
    evaluate.py  -- evaluation utilities, not reviewed in this design-focused
        conversation; check directly before relying on its current state
src/losses/
    tv_loss.py, lidar_loss.py  -- Phase 6 losses, stubs per the roadmap's still
        unchecked exit checklist, not yet implemented/verified
src/models/stage_b_temporal/  -- trimmed after the GF3D-faithful pivot, only what's
        still directly reused:
    buffer.py  -- GaussianState, ReferenceBuffer, recursive, architecture-independent
    current_frame_encoder.py  -- CurrentFrameEncoder, reused verbatim by the new design
    deform_heads.py  -- DeformHeadMu, DeformHeadR, quaternion/update-rule utilities
        (compute_relative_transform, etc.), reused verbatim
    the package init file
src/training/
    stage_b_engine.py  -- Motion-HyperNet-era orchestration; needs a real rewrite for
        the GF3D-faithful design, not yet started
    train_stage2.py  -- Phase 7 (joint fine-tune) stub, unstarted
scripts/
    Full-dataset infrastructure, current and active:
        check_vram_gs25600.py, generate_depth_gt_full.py, make_tiered_infos.py,
        make_frame0_infos.py
    Mini-dataset infrastructure, kept deliberately as a fast iteration tool:
        build_frame_index.py, run_stage_a_frame0.py, train_stage_a.py,
        train_stage1.py, train_stage1_smoketest.py, generate_depth_gt.py
    Generic diagnostics, architecture-independent, still valid:
        check_ego_compensation.py, measure_ego_motion_distribution.py,
        measure_gt_motion.py, measure_noise_floor.py, visualize_gaussians.py,
        evaluate_stage_a_all_scenes.py
    deprecated/  -- jobs fully superseded by later real results, kept for history:
        evaluate_heldout_scene.py, overfit_stage_a_single_frame.py,
        spot_check_dataloader.py, verify_scene_coverage.py
tests/
    test_stage_b_skeleton.py  -- trimmed to its architecture-independent
        quaternion-composition test; recursion/shape test deferred until the new
        deformation module exists to test against
    test_phase5_real_wiring.py  -- tests the Motion-HyperNet-era stage_b_engine.py
        wiring; will need revisiting once that file is rewritten for the new design
experiments/
    phase1_frame_index.json
    phase2_gaussian_viz/  -- per-scene position/scale/opacity plots

Full-scale checkpoints and the G_0 cache live outside this repo, on
/media/user/1TSSD/, alongside the separate GaussianFormer3D repo -- see
EXPERIMENT_LOG.md 2026-08-21 onward for the complete setup.

## Current assigned defaults (Stage A full-scale + Stage B GF3D-faithful)

These reflect the ACTIVE configuration, superseding the mini-dataset/Motion-HyperNet
era values (fully preserved in EXPERIMENT_LOG.md's earlier entries and on
archive/motion-hypernet-backup, not repeated here).

**Stage A (full-scale, trained)**

| Decision | Value | Rationale (short) |
|---|---|---|
| Num. Gaussians (N_g) | 25,600 | GaussianFormer3D's own real full-scale config; confirmed via a real VRAM check (~4.07GB forward pass) before committing to it |
| Scene set | nuScenes v1.0-trainval, 700 train / 150 val (850 total) | Real full-scale data, confirmed via the devkit's own reported totals matching our generated info files exactly |
| Camera backbone | ResNet101-DCN | Same as GF3D's own real config; unchanged from the earlier mini-dataset choice |
| Image resolution | 1600x900 raw, padded to 1600x928 | Confirmed: GF3D's real pipeline pads height to the next multiple of 32; the config's stated 900 is pre-padding |
| pc_range | [-50,-50,-5, 50,50,3] | GF3D's own real full-scale config value; wider than the mini-dataset's [-40,-40,-1, 40,40,5.4] |
| Depth source | Self-generated ground-truth depth (DepthHead_GTDpt) | GaussianFormer3D's own SharePoint depth-GT download link is dead; regenerated ourselves via real LiDAR-to-camera projection, reusing this project's own earlier depth-GT generator |
| Optimizer / LR | AdamW, 1e-4 | Matches GF3D's own real full-scale config |
| Epochs trained | 3 of a planned 6 (GF3D's own default is 24) | Deliberately stopped given the 20-day total time budget; gains were still real and ongoing (not plateaued) when stopped -- a time tradeoff, not a claim of convergence |
| Val subset for periodic eval | 70 of 150 scenes | Keeps per-epoch validation cost proportional; training itself uses the full 700 scenes |
| OOM mitigation | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` | Added after a real, recurring OOM found during training; reduces fragmentation-driven failures, does not reduce genuine peak memory need |
| Result | `mIoU=23.61` (epoch 3), real GF3D metric | ~87% of GaussianFormer3D's own reported 27.1, reached in far fewer epochs |

**Stage B (GF3D-faithful, design finalized, implementation not yet started)**

| Decision | Value | Rationale (short) |
|---|---|---|
| Core mechanism | GaussianFormer3D's own `DeformableFeatureAggregation3D`, reused directly | Not a custom-built attention mechanism; matches the reference architecture exactly |
| Query (Q) source | The reference buffer's current Gaussian state | Corrected during design review: GF3D's own frozen, scene-independent Q parameter can't carry forward recursively across frames the way a buffer-derived Q can |
| LiDAR for t>0 | Included | Unlike the earlier VGGT design, which dropped it for simplicity |
| K/V source | Frame t only (no explicit two-frame differencing) | A deliberate, disclosed simplification -- temporal signal still enters via the query/anchor mismatch, not an explicit diff term |
| Sparse-conv self-interaction | Not included | Disclosed gap vs. GF3D's own per-block structure; defensible given this design updates an existing state rather than constructing one from scratch |
| Deformation heads | `DeformHeadMu`/`DeformHeadR`, unchanged | Same residual, tanh-bounded heads used across every architecture this project has tried |
| Block iteration scheme | **Not yet finalized** -- Design A (single final commit) vs. Design B (cascaded, matching GF3D's own within-frame convergence) | Design A requires an additional, not-yet-empirically-validated fix (Section 3.6) to bridge a real mismatch between what grounds the search and what gets committed; Design B avoids this but was not the original choice. Decide before implementation begins. |

Full architecture, method, math, and the complete professor Q&A:
`docs/STAGE_B_GF3D_FAITHFUL_DESIGN.md`.
