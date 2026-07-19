# Occ4DGS: Dynamic 4D Gaussian Splatting for Occupancy Prediction in Autonomous Driving

Feedforward temporal deformation of GaussianFormer3D primitives for dynamic 3D semantic
occupancy prediction, on nuScenes v1.0-mini (10 scenes) with Occ3D-nuScenes GT.

CCU Autonomous Driving Perception Lab, advised by Prof. Rachael (Jui-Chiu) Chiang.

See `docs/IMPLEMENTATION_ROADMAP.md` for the full phase-by-phase plan and exit checklists,
`EXPERIMENT_LOG.md` for the running research log, and `docs/design_doc_v2.md` +
`docs/dataset_compute_addendum.md` for the architecture and data-source rationale.

**Status (as of Phase 2 completion):** Stage A (GaussianFormer3D, reused directly rather
than reimplemented — see note below) reproduces cleanly on all 10 scenes; full training
path validated on a single-frame overfit (loss 26.70→21.62 over 200 iters, reference
mIoU 0.1366). Phase 3 as originally scoped is subsumed by this result. Next: Phase 4
(Stage B skeleton).

## Architecture note: reuse, not reimplementation

The original plan (see `docs/design_doc_v2.md`) assumed building `src/models/stage_a_gaussianformer3d/`
from scratch. In practice, GaussianFormer3D's own `BEVSegmentorLiDAR3D` class turned out to
be directly reusable — we write a config (`configs/occ4dgs_mini_occ3d_gs6400.py`) and a thin
dataset adapter (`src/datasets/occ4dgs_dataset.py`) that feeds our Occ3D-mini data into their
real pipeline, rather than porting their architecture ourselves. This was a deliberate pivot
made during Phase 2 once their code was confirmed importable and correct (Phase 0) — see
`EXPERIMENT_LOG.md` 2026-07-18/19 entries for the full derivation and every bug found along
the way (calibration math, config completeness, environment leaks).

## Assigned defaults (decided, not pending professor approval)

These were previously open decisions; they are now fixed as working defaults and adjusted
only via pilot runs, not left unresolved. **Two of these changed from the original plan
once real numbers were in hand — both logged explicitly, not silent drift:**

| Decision | Value | Rationale (short) |
|---|---|---|
| `N_g` (num. Gaussians) | **6,400** | Fits single RTX 3090 24GB with Stage B unroll; matches GaussianFormer-2's own ablation showing modest IoU cost vs 25,600 at ~4x less memory. Confirmed: Phase 2 forward pass peaked at only 2.84GB, so 12,800 remains a live option to revisit post-Phase 8 if quality needs it. |
| Camera backbone | **ResNet101-DCN** *(changed from originally planned ResNet50)* | Reasoning at decision time: with only 10 scenes, pretrained-checkpoint quality matters more than parameter count, and reusing GaussianFormer3D's own tested `r101_dcn_fcos3d_pretrain.pth` checkpoint avoids introducing a second unverified variable alongside the new dataset adapter. Confirmed cheap in practice: 2.84GB peak VRAM, well within budget — ResNet50 was never actually needed as a memory-saving fallback. |
| Image resolution | **900×1600 (padded to 928×1600), full resolution** *(changed from originally planned 450×800 downscale)* | Same reasoning as backbone: VRAM was never the constraint it was assumed to be (2.84GB peak vs. 24GB available). No downscaling needed; revisit only if Stage B's multi-frame unroll changes the VRAM picture. |
| Stage 1 (frozen warm-up) LR | **1e-4** (AdamW, cosine schedule) for HyperNet + Φ_μ + Φ_r | Matches GaussianFormer3D's own nuScenes LR for new modules. Not yet exercised — Stage B doesn't exist yet (Phase 4+). |
| Stage 2 (joint fine-tune) LR | **1e-5** for Stage A (GaussianFormer3D) params, **5e-5** for temporal module | 10x lower LR for the already-converged generator; ratio, not absolute value, is what matters. Not yet exercised. |
| Unroll window (Stage 1) | **2 frames** | Conservative starting point for VRAM; to be confirmed via Phase 5 profiling. |
| Unroll window (Stage 2) | **3 frames** | Increased once Stage 1 stability confirmed. |
| Epochs | Stage 1: **60**, Stage 2: **40** | Small dataset (10 scenes, ~400 frames) — epochs are cheap; adjust based on validation curve, not fixed in stone. |
| Batch size | **1 sequence/step**, gradient accumulation ×4 | Effective batch 4 on single GPU. |
| Precision | **AMP (fp16)** | Near-mandatory on 24GB in principle; not yet stress-tested since Stage A alone runs comfortably without it (Phase 2's 2.84GB peak was full fp32). Revisit once Stage B's multi-frame unroll is in place. |
| Frozen vs. joint | **Staged**: Stage 1 frozen Stage A → Stage 2 joint fine-tune | Already argued in `design_doc_v2.md` §5; kept as main config, frozen-only kept as ablation. |
| GT source | **Occ3D-nuScenes** | Existing verified loader from 3DGS project; see `dataset_compute_addendum.md`. Confirmed in Phase 0/1: 100% GT coverage across all 10 scenes (two on-disk layouts, both handled — see `EXPERIMENT_LOG.md`). |
| Scene set | **nuScenes v1.0-mini (10 scenes)** | Self-contained, matches professor's scope exactly. |
| Occ3D coordinate mapping | **`occ_annotation="occ3d"`** in `GaussianLifterLiDAR` | GaussianFormer3D already ships this exact branch (`pc_range_3` in `safe_ops.py`, matching our range verbatim) — discovered during Phase 2, not something we needed to add. |

All of the above are recorded here so any future run's config diverging from this table is a
deliberate, logged decision (see `EXPERIMENT_LOG.md`), not an accidental default drift.

## Repo layout (as it actually exists post-Phase 2)

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
                                    # separate src/models/stage_a_gaussianformer3d/ module
scripts/
  build_frame_index.py            # Phase 1
  spot_check_dataloader.py        # Phase 1
  generate_depth_gt.py            # Phase 2 -- our own depth_gt generator, replacing
                                    # GaussianFormer3D's SharePoint-hosted download
  run_stage_a_frame0.py           # Phase 2 -- first successful forward pass
  visualize_gaussians.py          # Phase 2 -- collapse/saturation check, all 10 scenes
  overfit_stage_a_single_frame.py # Phase 2 -- training path validation + reference mIoU
experiments/
  phase1_frame_index.json         # source of truth for has_gt-tagged frames, all phases
  phase2_gaussian_viz/            # per-scene position/scale/opacity plots
```