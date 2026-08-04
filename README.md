# Occ4DGS: Dynamic 4D Gaussian Splatting for Occupancy Prediction in Autonomous Driving

Feedforward temporal deformation of GaussianFormer3D primitives for dynamic 3D semantic
occupancy prediction, on nuScenes v1.0-mini (10 scenes) with Occ3D-nuScenes GT.

CCU Autonomous Driving Perception Lab, advised by Prof. Rachael (Jui-Chiu) Chiang.

See `docs/IMPLEMENTATION_ROADMAP.md` for the full phase-by-phase plan and exit checklists,
`EXPERIMENT_LOG.md` for the running research log, and `docs/design_doc_v2.md` +
`docs/dataset_compute_addendum.md` for the architecture and data-source rationale.

**Status (as of Phase 5, paused pending noise-floor measurement):** Stage A is now
**really trained** (not just the single-frame overfit reported at Phase 2) — 8 of the 10
mini-set scenes, with `scene-1094`/`scene-1100` fixed as a **permanent, never-trained-on
validation pair**. Stage B's temporal deformation mechanism (`HyperNet`/`ConvHyperNet` →
per-Gaussian grid query → `Φ_μ`,`Φ_r` → update rule) is fully wired against real encoders
and trains real, positive in-sample signal, but has **not yet demonstrated reliable
held-out generalization**. A long debugging/architecture-refinement arc (ego-motion
compensation, a discovered Gaussian-budget representational gap, four architecture
refinements informed by direct review of the 4DGC reference implementation) each produced
modest, real single-run improvements — but a late discovery that the deterministic
"do-nothing" baseline itself varies by ~0.036 run-to-run (same order of magnitude as every
claimed improvement) means **all of these comparisons are currently provisional**. Work is
paused pending a repeated-seed noise-floor measurement before any further architecture
conclusions are drawn. Full arc: `EXPERIMENT_LOG.md`, entries 2026-07-27 through 2026-07-30.

## Architecture note: reuse, not reimplementation

The original plan (see `docs/design_doc_v2.md`) assumed building `src/models/stage_a_gaussianformer3d/`
from scratch. In practice, GaussianFormer3D's own `BEVSegmentorLiDAR3D` class turned out to
be directly reusable — we write a config (`configs/occ4dgs_mini_occ3d_gs6400.py`) and a thin
dataset adapter (`src/datasets/occ4dgs_dataset.py`) that feeds our Occ3D-mini data into their
real pipeline, rather than porting their architecture ourselves. This was a deliberate pivot
made during Phase 2 once their code was confirmed importable and correct (Phase 0) — see
`EXPERIMENT_LOG.md` 2026-07-18/19 entries for the full derivation and every bug found along
the way (calibration math, config completeness, environment leaks).

Stage B's temporal module (`src/models/stage_b_temporal/`), by contrast, **is** new,
from-scratch code — only its encoder (frozen `img_backbone`/`img_neck`/`pts_dpt_head`,
reused from Stage A per the design doc's §2.2) is inherited, not reimplemented.

## Assigned defaults (decided, not pending professor approval)

These were previously open decisions; they are now fixed as working defaults and adjusted
only via pilot runs, not left unresolved. **Several of these changed from the original plan
once real numbers were in hand — all logged explicitly, not silent drift:**

| Decision | Value | Rationale (short) |
|---|---|---|
| `N_g` (num. Gaussians) | **6,400** | Fits single RTX 3090 24GB with Stage B unroll; matches GaussianFormer-2's own ablation showing modest IoU cost vs 25,600 at ~4x less memory. Confirmed: Phase 2 forward pass peaked at only 2.84GB, so 12,800 remains a live option to revisit post-Phase 8 if quality needs it. Held throughout Phase 5's entire architecture-refinement arc — never needed revisiting. |
| Camera backbone | **ResNet101-DCN** *(changed from originally planned ResNet50)* | Reasoning at decision time: with only 10 scenes, pretrained-checkpoint quality matters more than parameter count, and reusing GaussianFormer3D's own tested `r101_dcn_fcos3d_pretrain.pth` checkpoint avoids introducing a second unverified variable alongside the new dataset adapter. Confirmed cheap in practice: 2.84GB peak VRAM, well within budget — ResNet50 was never actually needed as a memory-saving fallback. |
| Image resolution | **900×1600 (padded to 928×1600), full resolution** *(changed from originally planned 450×800 downscale)* | Same reasoning as backbone: VRAM was never the constraint it was assumed to be (2.84GB peak vs. 24GB available). No downscaling needed; Stage B's 2-frame unroll training runs comfortably at 3.2-3.9GB peak, confirming this held under real Phase 5 load, not just Stage A alone. |
| Stage 1 (frozen warm-up) LR | **1e-4** (AdamW, cosine schedule) for HyperNet + Φ_μ + Φ_r | Matches GaussianFormer3D's own nuScenes LR for new modules. **Confirmed in practice, Phase 5** — used unchanged across every Stage B training run this session (Steps 1-4, the scene-scaling sweep), no retuning needed. |
| Stage 1 weight decay / dropout / motion penalty | **0.05 / 0.2 / 0.01** *(new, not in original plan)* | Added as a regularization pass once held-out generalization first proved elusive. **Result: no measurable effect on held-out generalization** (EXPERIMENT_LOG.md 2026-07-30 retrospective) — kept in the code (harmless, cheap) but not the driver of any of the real improvements found afterward. Worth retesting only once the data-scale question is separately resolved (see Phase 5 status below), not before. |
| Stage 2 (joint fine-tune) LR | **1e-5** for Stage A (GaussianFormer3D) params, **5e-5** for temporal module | 10x lower LR for the already-converged generator; ratio, not absolute value, is what matters. **Still not yet exercised** — all Phase 5 work so far has been Stage 1 (frozen Stage A) only; Stage 2 remains Phase 7. |
| Unroll window (Stage 1) | **2 frames** | Confirmed in practice across all Phase 5 training — no VRAM pressure observed at this window (peak 3.2-3.9GB, comfortably under budget). |
| Unroll window (Stage 2) | **3 frames** | Increased once Stage 1 stability confirmed. **Still not yet exercised** — Stage 2 pending. |
| Epochs | Stage 1: **20** *(reduced from originally planned 60)*, Stage 2: **40** | Reduced deliberately during Phase 5's rapid architecture-iteration phase — 20 epochs at 8 scenes gives ~1580 optimizer steps, enough to see the full collapse-then-plateau pattern that turned out to be the dominant signal every run. **Worth revisiting for a final, non-iterative training run** once the noise-floor question (see status below) is resolved — 20 was chosen for iteration speed, not confirmed as sufficient for a final result. |
| Batch size | **1 sequence/step**, gradient accumulation ×4 | Effective batch 4 on single GPU. Confirmed unchanged throughout Phase 5. |
| Precision | **fp32, no AMP** | Still not stress-tested with AMP — Stage B's 2-frame unroll training has run comfortably at 3.2-3.9GB peak without it, so the "near-mandatory" assumption from the original plan has not yet been tested in practice. |
| Frozen vs. joint | **Staged**: Stage 1 frozen Stage A → Stage 2 joint fine-tune | Already argued in `design_doc_v2.md` §5; kept as main config, frozen-only kept as ablation. **All Phase 5 work to date is Stage 1 (frozen) only** — Stage 2 (joint) is unstarted. |
| GT source | **Occ3D-nuScenes** | Existing verified loader from 3DGS project; see `dataset_compute_addendum.md`. Confirmed in Phase 0/1: 100% GT coverage across all 10 scenes (two on-disk layouts, both handled — see `EXPERIMENT_LOG.md`). |
| Scene set | **nuScenes v1.0-mini (10 scenes)** | Self-contained, matches professor's scope exactly. |
| **Stage A train/held-out split** *(new, decided during Phase 5)* | **8 scenes trained** (`scene-0061, 0103, 0553, 0655, 0757, 0796, 0916, 1077`), **2 scenes permanently held out** (`scene-1094, scene-1100`) | Chosen via a neutral rule (next scenes in existing list order), not cherry-picked. This 8/2 split is now fixed and reused as the constant validation set for every Stage A and Stage B comparison this session — never trained on by any checkpoint, to keep every held-out comparison genuinely fair. |
| **Random seed** *(new, added Phase 5)* | **42**, fixed | Added after the original 1/3/6/8 scene-scaling sweep showed unexplained run-to-run variance. **Important caveat discovered later:** fixing the seed did NOT eliminate all non-determinism — the do-nothing baseline (which should be perfectly deterministic given a fixed, frozen, already-trained Stage A checkpoint) still varied by ~0.036 across otherwise-identical runs. Root cause not yet identified (likely GPU-level non-deterministic kernels); flagged as an open, unresolved measurement-precision issue — see Phase 5 status below. |
| **Grid resolution / channel depth** *(new, decided during Phase 5's Step 2)* | Resolutions **(8, 16, 32)**, `grid_feat_dim=4` per level *(changed from initial (4,8,16) / 16)* | Rebalanced toward finer spatial resolution with fewer channels per cell, matching 4DGC's own tradeoff (confirmed via direct reference-code review) rather than our original coarse-grid/deep-channel guess. Produced a real, single-run improvement (best held-out delta +0.044→+0.080) — but see the noise-floor caveat above before treating this as fully confirmed. |
| **Motion-grid query mechanism** *(resolved during Phase 5's Step 3)* | Each grid level sampled via `sin(2^l·π·x)`/`cos(2^l·π·x)` as the grid-sample coordinates themselves | Resolves an ambiguity flagged since Phase 4 in `design_doc_v2.md` §2.4's notation. Confirmed via direct reading of 4DGC's actual source (`Motion_Grid.interpolate()`/`positional_encoding()`), not inference — this project's earlier guess (position as coordinate, PE concatenated as separate context) is kept as `query_motion_grid`'s default for backward compatibility, with the confirmed-correct mechanism implemented separately as `query_motion_grid_pe_coordinate`. | 
| Occ3D coordinate mapping | **`occ_annotation="occ3d"`** in `GaussianLifterLiDAR` | GaussianFormer3D already ships this exact branch (`pc_range_3` in `safe_ops.py`, matching our range verbatim) — discovered during Phase 2, not something we needed to add. |

All of the above are recorded here so any future run's config diverging from this table is a
deliberate, logged decision (see `EXPERIMENT_LOG.md`), not an accidental default drift.

## Phase 5 status in plain terms

- **Stage A**: trained for real, working, at a confirmed data-scale ceiling (gap between
  trained and held-out scenes narrowed with more scenes but never closed — expected, given
  8-10 scenes vs. the hundreds a full nuScenes training set would provide).
- **Stage B mechanism**: confirmed to work — learns real, positive in-sample signal
  consistently, responds predictably to targeted fixes (e.g. delta-conditioning cleanly
  fixed a confirmed scene-recognition shortcut), and ego-motion compensation's own geometry
  was independently verified correct (a near-zero-real-motion scene showed no damage from
  applying it).
- **Stage B held-out generalization**: not yet reliably achieved. A real, non-bug
  architectural gap was found and characterized (a fixed 6400-Gaussian budget has no way
  to represent newly-visible content after real ego motion — confirmed to be the *majority*
  regime, not an edge case, at 44.4% of all measured clips). A first fix attempt (learned
  `SpawnHead`) gave a small real improvement but did not close the gap.
- **Four architecture refinements** (informed directly by reading 4DGC's actual reference
  code, not assumption): rotation-composition order, grid resolution/channel rebalance,
  the real PE-as-coordinate query mechanism, and a spatially-structured convolutional
  `HyperNet` decoder. Each produced a modest, real, single-run improvement in the best-case
  held-out result, and the fourth (the conv decoder) was the first to delay — not just
  raise the peak of — the collapse-then-plateau pattern every version has shown.
- **Open, unresolved issue, current blocker:** the deterministic do-nothing baseline itself
  varies by ~0.036 run-to-run — the same order of magnitude as every improvement claimed
  above. **All architecture comparisons this session are therefore provisional** until a
  repeated-seed noise floor is measured. This is the next planned step, not yet run.

## Repo layout (as it actually exists, Phase 5)

```
configs/
  occ4dgs_mini_occ3d_gs6400.py   # the real, working Stage A config (mirrored from
                                  # GaussianFormer3D/config/, kept in sync manually --
                                  # see GIT_WORKFLOW.md note on re-syncing after edits)
  dataset_mini_occ3d.yaml
src/datasets/
  nuscenes_mini.py                # scene/frame indexing via nuscenes-devkit
  occ3d_gt.py                     # Occ3D GT loader, handles both on-disk layouts
  occ4dgs_dataset.py               # the real dataset adapter feeding GaussianFormer3D's
                                    # pipeline -- this is "Stage A" in practice, not a
                                    # separate src/models/stage_a_gaussianformer3d/ module.
                                    # Also passes ego2global/lidar2global through (Phase 5,
                                    # added for ego-motion compensation).
  occ4dgs_clip_dataset.py          # Phase 4 -- Occ4DGSClipDataset, wraps Occ4DGSDataset
                                    # into temporally-adjacent frame pairs for Stage B.
src/models/stage_b_temporal/       # Phase 4/5 -- Stage B's temporal deformation module,
                                    # genuinely new code (unlike Stage A's reuse pattern)
  buffer.py                       # GaussianState, ReferenceBuffer (recursive, no
                                    # re-anchoring to G_0)
  hypernet.py                     # MotionHyperNet -- original per-level Linear decoder
                                    # design (kept for reference/comparison)
  conv_hypernet.py                # ConvHyperNet -- Phase 5 Step 4 replacement, a shared
                                    # transposed-convolutional decoder (<1M params,
                                    # spatially-structured), current default
  grid_query.py                   # query_motion_grid (original guess, kept as default)
                                    # and query_motion_grid_pe_coordinate (Phase 5 Step 3,
                                    # 4DGC's actual confirmed mechanism)
  deform_heads.py                 # DeformHeadMu, DeformHeadR, apply_update_rule,
                                    # ego-motion compensation utilities
                                    # (rotmat_to_quat, compute_relative_transform,
                                    # apply_ego_compensated_update_rule), quaternion utils
  spawn_head.py                   # SpawnHead -- Phase 5 learned recycling mechanism for
                                    # Gaussians pushed out of pc_range by compensation
  pool_features.py                # PoolFeatures -- global-average-pools multi-scale
                                    # encoder features into a single pooled vector
  current_frame_encoder.py        # CurrentFrameEncoder -- wraps Stage A's frozen
                                    # img_backbone/img_neck/pts_dpt_head for per-frame reuse
  __init__.py
scripts/
  build_frame_index.py            # Phase 1
  spot_check_dataloader.py        # Phase 1
  generate_depth_gt.py            # Phase 2 -- our own depth_gt generator, replacing
                                    # GaussianFormer3D's SharePoint-hosted download
  run_stage_a_frame0.py           # Phase 2 -- first successful forward pass; also now
                                    # provides to_batch_of_one/build_pipeline, reused
                                    # throughout Phase 4/5's Stage B scripts
  visualize_gaussians.py          # Phase 2 -- collapse/saturation check, all 10 scenes
  overfit_stage_a_single_frame.py # Phase 2 -- training path validation + reference mIoU
  train_stage_a.py                # Phase 5 -- real Stage A training (8/10 scenes)
  evaluate_stage_a_all_scenes.py  # Phase 5 -- Stage A-only eval across all 10 scenes,
                                    # adjusted mIoU (excludes zero-GT-support classes)
  train_stage1.py                 # Phase 4/5 -- the real Stage B training loop; also
                                    # contains deform_one_step, evaluate_heldout, and all
                                    # ego-motion-compensation/architecture-variant wiring
  train_stage1_smoketest.py       # Phase 4 -- single-repeated-clip sanity check
  evaluate_heldout_scene.py       # Phase 5 -- do-nothing vs. trained comparison on the
                                    # permanent held-out pair
  measure_gt_motion.py            # Phase 5 -- measures real ground-truth occupancy
                                    # change between adjacent frames (found: 60-84%,
                                    # dominated by ego motion)
  measure_ego_motion_distribution.py  # Phase 5 -- measures real ego-motion magnitude
                                    # distribution across all 394 clips (found: 44.4%
                                    # exceed the damage threshold -- majority regime)
  check_ego_compensation.py       # Phase 5 -- isolates ego-motion compensation's
                                    # geometric correctness independent of any learning
tests/
  test_stage_b_skeleton.py        # Phase 4 exit-checklist suite (quaternion composition,
                                    # grid-sample coordinate convention, 2-frame toy
                                    # recursion) -- regression-checked after every
                                    # architecture change this session, never broken
experiments/
  phase1_frame_index.json         # source of truth for has_gt-tagged frames, all phases
  phase2_gaussian_viz/            # per-scene position/scale/opacity plots
  stage_a_checkpoints/            # Phase 5 -- trained Stage A weights (stage_a_best.pth)
  stage_b_temporal_checkpoints/   # Phase 5 -- trained Stage B temporal module weights,
                                    # per scene-count (stage1_warmup_temporal_nN.pth)
```