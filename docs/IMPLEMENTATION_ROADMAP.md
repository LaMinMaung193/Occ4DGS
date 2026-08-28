# Implementation Roadmap — Occ4DGS

Each phase has: **Goal**, **Steps**, **Config used**, **Deliverables**, **Exit checklist**,
**Git tag**. Do not start phase N+1 until phase N's exit checklist is fully checked — this
mirrors the discipline that worked in QG-Fusion (EXPERIMENT_LOG.md / ROADMAP.md source-of-truth
pattern). Log every run, pass or fail, in `EXPERIMENT_LOG.md` using the template in
`docs/EXPERIMENT_LOG_TEMPLATE.md`.

**Architecture note (read before Phase 2+):** the original plan assumed building Stage A as
our own module (`src/models/stage_a_gaussianformer3d/`). In practice we reuse GaussianFormer3D's
`BEVSegmentorLiDAR3D` directly via a config + dataset adapter instead — see `README.md`'s
"Architecture note" section and `EXPERIMENT_LOG.md`'s 2026-07-18/19 entries. Phase 2's
"Deliverables" below reflect what was actually built, not the original module-path plan.

**This roadmap now tracks the GF3D-faithful design specifically** (`docs/STAGE_B_GF3D_FAITHFUL_DESIGN.md`),
following a professor-directed architecture pivot. Full history of what led to that pivot lives
in `EXPERIMENT_LOG.md`; this document stays forward-focused on the active design.

---

## Phase 0 — Environment, repo, and data verification ✓ COMPLETE

**Goal:** confirm the ground is solid before writing any model code.

**Steps:**
1. `git init` this repo (see `GIT_WORKFLOW.md`), push empty skeleton to GitHub remote.
2. Clone GaussianFormer3D's public repo into a scratch venv; attempt to run its own
   demo/inference script. Record every dependency conflict against your existing
   `Python 3.8 / CUDA 12.8 / spconv 2.3.6` stack.
3. Freeze the resolved working versions into `requirements.txt` (replace the `TBD` placeholders).
4. Verify all 10 `v1.0-mini` scene names exist as folders in `data/occ3d/gts/` (§2 of
   `dataset_compute_addendum.md`). Write this as `scripts/verify_scene_coverage.py`.
5. Confirm `pc_range`/voxel size alignment between your intended Stage A voxelization and the
   Occ3D grid (`configs/dataset_mini_occ3d.yaml`).
6. Symlink `data/nuscenes_mini -> /media/user/Transcend/nuScenes/v1.0-mini` and
   `data/occ3d_gts -> /media/user/Transcend/data/occ3d/gts` inside the repo's `data/` folder
   (gitignored, but keeps all paths repo-relative in code).

**Config used:** `configs/dataset_mini_occ3d.yaml` (read-only verification, no training yet).

**Deliverables:** working `gf3d` conda env, `scripts/verify_scene_coverage.py`, resolved
`requirements.txt`, symlinked `data/`.

**Exit checklist:**
- [x] GaussianFormer3D repo's core imports (mmdet3d/mmcv/spconv/DFA3D/LocalAggregator/
      GaussianOccEncoder3D) succeed cleanly; `unittest_DFA3D.py` passes. **Note:** full
      `eval.py`/`train.py` run against author-provided weights+data was deliberately NOT
      performed — not required, since Occ4DGS writes its own Stage A entry point rather
      than reusing their pkl-based dataloader (checklist wording corrected from the
      original "runs its own demo end-to-end" to match what was actually verified).
- [x] `requirements.txt` has concrete, tested version numbers (no `TBD` left)
- [x] All 10 mini scene names confirmed present in `data/occ3d/gts/`
- [x] `pc_range` / voxel size match verified in writing (see EXPERIMENT_LOG.md
      2026-07-14-pc-range-verification)
- [x] Repo pushed to GitHub with this exit state tagged

**Git tag:** `v0.0-phase0-env-verified`

---

## Phase 1 — Frame index & data loading ✓ COMPLETE

**Goal:** a reliable per-scene frame index that knows which frames have valid GT, before any
model touches the data.

**Steps:**
1. Build `src/datasets/nuscenes_mini.py`: loads the 10-scene nuscenes-devkit tables, returns
   ordered per-scene frame lists with sample tokens.
2. Build `src/datasets/occ3d_gt.py`: given a sample token, loads the Occ3D voxel label
   (reuse loader logic from the 3DGS project); returns `None`/flag if missing.
3. Tag every frame `has_gt: bool`; write out `experiments/phase1_frame_index.json` — this is
   the single source of truth every later phase's dataloader reads from.
4. Slice contiguous valid-GT runs per scene; log the resulting run-length distribution.

**Config used:** `configs/dataset_mini_occ3d.yaml`.

**Deliverables:** `src/datasets/nuscenes_mini.py`, `src/datasets/occ3d_gt.py`,
`experiments/phase1_frame_index.json`, `scripts/build_frame_index.py`,
`scripts/spot_check_dataloader.py`.

**Exit checklist:**
- [x] Frame index built for all 10 scenes — **result was better than expected: 100% GT
      coverage on every scene**, not the partial coverage the "index-39 gap" from the prior
      3DGS project suggested. Root cause of an initial false-negative (4/10 scenes reading
      as 0%) traced to a dual on-disk layout convention, not a real data gap — see
      EXPERIMENT_LOG.md 2026-07-14-frame-index-and-gt-loader.
- [x] Every scene has a contiguous valid-GT run ≥ 3 frames — all 10 scenes have runs of
      their full length (39-41 frames), far exceeding the minimum.
- [x] Dataloader returns correctly-shaped camera tensors, LiDAR points, and GT voxels,
      spot-checked on all 10/10 scenes (not just 1).

**Git tag:** `v0.1-phase1-data-index`

---

## Phase 2 — Stage A reproduction (via reuse, not reimplementation) ✓ COMPLETE

**Goal:** GaussianFormer3D producing sane Gaussians on your machine, with a working full
training path, before Stage B exists.

**Steps actually taken** (see architecture note at top — this diverged from the original
plan of building a standalone `src/models/stage_a_gaussianformer3d/` module):
1. Wrote `configs/occ4dgs_mini_occ3d_gs6400.py` — a from-scratch config (no `_base_`
   inheritance, deliberately, to avoid silently importing unverified SurroundOcc/2D
   defaults) for GaussianFormer3D's real `BEVSegmentorLiDAR3D`, with `N_g=6400`,
   `occ_annotation="occ3d"`, ResNet101-DCN backbone.
2. Wrote `src/datasets/occ4dgs_dataset.py` — builds the exact `input_dict` shape their
   pipeline expects, sourcing frames from `nuscenes_mini.py`/`occ3d_gt.py` instead of
   their external pkl. Includes `lidar_path`, `sweeps` (ported `obtain_sensor2top` logic),
   and dual-layout-aware `occ_path` resolution.
3. Wrote `scripts/generate_depth_gt.py` — our own LiDAR-to-camera depth projection,
   replacing GaussianFormer3D's SharePoint-hosted `depth_gt/` download (not available for
   our Occ3D-mini setup). Validated against real point counts/pixel ranges after fixing an
   initial `cam2img`/`img2cam` inversion bug.
4. Wrote `scripts/run_stage_a_frame0.py` — ran the full forward pass on one scene's frame 0.
   Resolved 11 distinct bugs to get here (missing dict keys, `_delete_=True` config-merge
   artifacts, silently-defaulted falsy config fields, an in-place tensor mutation) — full
   list in EXPERIMENT_LOG.md 2026-07-18 and 2026-07-19 entries.
5. Wrote `scripts/visualize_gaussians.py` — position scatter + scale/opacity histograms,
   automatic collapse/saturation detection, run across all 10 scenes.
6. Wrote `scripts/overfit_stage_a_single_frame.py` — validated the full training path
   (loss → backward → gradient clip → optimizer step) and produced a reference mIoU.

**Config used:** `configs/occ4dgs_mini_occ3d_gs6400.py`.

**Deliverables (actual, not original plan):**
- `configs/occ4dgs_mini_occ3d_gs6400.py`
- `src/datasets/occ4dgs_dataset.py`
- `scripts/generate_depth_gt.py`, `scripts/run_stage_a_frame0.py`,
  `scripts/visualize_gaussians.py`, `scripts/overfit_stage_a_single_frame.py`
- `experiments/phase2_gaussian_viz/*.png` — one plot per scene, all 10 scenes
- Peak VRAM: **2.84 GB** (single sample, full 900×1600 resolution, ResNet101-DCN, no AMP)
- Reference mIoU: **0.1366** (single-frame, 200-iteration overfit, from-scratch encoder)

**Exit checklist:**
- [x] No position/Z-axis collapse, no scale saturation, on any of the 10 scenes' `G_0` —
      confirmed via `scripts/visualize_gaussians.py`'s automatic checks, all 10 scenes clean,
      remarkably consistent stats across scenes (expected, since initialization dominates
      over data differences before any training).
- [x] Peak VRAM for Stage A alone recorded: **2.84 GB**, far under the 24GB budget —
      substantial headroom confirmed for Stage B's later unroll.
- [x] Standalone frame-0 mIoU logged as a reference number: **0.1366** (checklist item
      upgraded from "optional" to "important" and completed — see README.md decision log).
      Loss curve (26.70→21.62 over 200 iters) is the primary pass/fail signal; the training
      path itself (loss computation, backward, gradient clipping, optimizer step) is
      confirmed working end-to-end, which was the real point of this check.

**Git tag:** `v0.2-phase2-stageA-reproduced`

---

## Phase 3 — Stage C wiring smoke test — **SUBSUMED BY PHASE 2, not separately executed**

**Original goal:** verify Gaussian-to-voxel splatting and the occupancy loss work end-to-end,
using Stage A's cached output, before Stage B introduces any new complexity.

**Why this phase didn't need separate execution:** the original plan assumed we'd build our
own `src/models/stage_c_splatting/` module and wire it to a cached, frozen `G_0`. Since Phase 2
instead reuses GaussianFormer3D's full `BEVSegmentorLiDAR3D` — whose `head` submodule already
*embeds* Gaussian-to-voxel splatting internally — this exact check was performed as part of
Phase 2's `overfit_stage_a_single_frame.py` run: the loss decreased over 200 iterations, which
is only possible if splatting, the occupancy loss, and gradient flow back into the Gaussian
parameters are all working correctly together.

**Exit checklist — mapped to Phase 2 evidence, not re-run separately:**
- [x] `Ô` shape matches Occ3D grid — confirmed: `pred_occ` shape `[1, 18, 640000]` = 18
      classes × (200×200×16) voxels, exactly as expected.
- [x] Loss decreases monotonically on the single-frame overfit test — confirmed: 26.70→21.62
      over 200 iterations (`EXPERIMENT_LOG.md` 2026-07-19).
- [x] Gradients confirmed flowing back into Stage A's Gaussian parameters — implied directly
      by the loss decrease (an unconnected graph cannot decrease loss via `optimizer.step()`);
      not separately verified via explicit `.grad` inspection, since the loss-curve evidence
      is stronger and was already in hand.

**Git tag:** none — no separate commit; covered by `v0.2-phase2-stageA-reproduced`.

---

## Phase 4 — Stage B skeleton (shape/recursion validation only) ✓ COMPLETE

**Goal:** validate the recursive buffer mechanics and tensor shapes with dummy encoders,
before wiring in real camera/LiDAR features.

**Steps:**
1. Implement `src/models/stage_b_temporal/buffer.py`: the reference buffer object
   (`read()`, `write(G_t)` — recursive, no re-anchoring). **Still directly reused by the
   current GF3D-faithful design, unchanged** — see `docs/STAGE_B_GF3D_FAITHFUL_DESIGN.md`.
2. Implement dummy (randomly initialized, untrained) encoder/head wiring for a 2-frame toy
   sequence, to validate mechanics before real features exist.
3. Wire: `buffer.read() → [deformation mechanism] → buffer.write(G_t) → Stage C splatting`,
   on a 2-frame toy sequence. **Note:** "Stage C splatting" here means calling into
   `BEVSegmentorLiDAR3D`'s embedded head (per Phase 2/3's architecture pivot), not a
   standalone module.
4. Confirm shapes at every step; confirm the buffer correctly holds `G_t` (not `G_0`) after
   one step, i.e. recursion is actually recursive and not silently re-reading `G_0`.

**Config used:** structure only, dummy inputs — no training.

**Deliverables:** `src/models/stage_b_temporal/buffer.py`, `deform_heads.py`,
`tests/test_stage_b_skeleton.py` (the quaternion-composition correctness test — kept and
reused as a standing regression check across every architecture this project has tried;
trimmed to its architecture-independent core after the GF3D-faithful pivot, since the
original recursion/shape test depended on since-removed, architecture-specific modules —
that test is deferred until the GF3D-faithful deformation module exists to test against).

**Exit checklist:**
- [x] Buffer state after step 1 is provably `G_1` (deformed), not `G_0` — asserted directly
      against `ReferenceBuffer.write_count`, not eyeballed.
- [x] All tensor shapes match across the full chain for a 2-frame toy sequence — confirmed.
- [x] Quaternion composition (`Δr_t ⊗ r_{t-1}`, normalized) verified numerically on a
      hand-computed example — confirmed, still passing after the GF3D-faithful pivot
      (`tests/test_stage_b_skeleton.py`).

**Git tag:** `v0.4-phase4-stageB-skeleton`

---

## Phase 5 — Real encoders + Stage 1 (frozen warm-up) training — **CONCLUDED (superseded)**

**Original goal (as scoped before this phase began):** wire in real Stage A encoders, train
the temporal deformation mechanism per a frozen-warm-up schedule, confirm the deformed frame
beats a "do-nothing" baseline, scale to all 10 scenes.

Substantial real work happened in this phase — full detail lives in `EXPERIMENT_LOG.md`'s
entries from 2026-07-27 onward.

### Current status: CONCLUDED (superseded)

This phase's own open item (a repeated-seed noise-floor measurement) was resolved, and
architecture work continued past this point — but the whole line was ultimately superseded
by a professor-directed pivot to the GF3D-faithful design. **This roadmap's phases from here
forward track that design specifically.** Full history between this point and the pivot:
`EXPERIMENT_LOG.md`. Current active architecture: `docs/STAGE_B_GF3D_FAITHFUL_DESIGN.md`.

**Deliverables (still relevant, reused going forward):** `scripts/measure_gt_motion.py`,
`scripts/measure_ego_motion_distribution.py`, `scripts/check_ego_compensation.py`
(architecture-independent diagnostic tooling); `src/models/stage_b_temporal/buffer.py`,
`deform_heads.py` (directly reused by the current design, unchanged).

**Git tag:** none — superseded before a stable tag point was reached.

---

## Phase 5B — Full-dataset Stage A ✓ COMPLETE

**Goal:** move Stage A from the ~8-10 scene mini dataset to full-scale nuScenes v1.0-trainval,
and produce a real, cached `G_0` for every scene, so no future Stage B work needs to re-run
Stage A.

**Steps:**
1. Confirmed `N_g=25,600` (GaussianFormer3D's own real full-scale config) fits in available
   VRAM via a synthetic-tensor forward-pass check, before committing to any real data work.
2. Wired the real, full-scale data pipeline: nuScenes v1.0-trainval + SurroundOcc symlinks,
   generated info files via GaussianFormer3D's own real tooling, self-generated depth-GT
   (204,894 files) after finding GaussianFormer3D's own download link dead.
3. Ran a small-tier (50-scene) gate check before committing to a full run — confirmed healthy,
   real held-out mIoU improvement, not just decreasing training loss.
4. Trained Stage A on the full 700-scene train split, `N_g=25,600`, stopped deliberately at
   epoch 3 of a planned 6 given the project's overall time budget.
5. Built and validated a `G_0` extraction pipeline; ran it across all 850 scenes (both splits
   combined), caching each scene's Stage A output to its own file.

**Config used:** GaussianFormer3D's own real `nuscenes_surroundocc_gs25600.py` (separate repo,
not this one) plus this project's own `scripts/check_vram_gs25600.py`,
`scripts/generate_depth_gt_full.py`, `scripts/make_tiered_infos.py`,
`scripts/make_frame0_infos.py`.

**Deliverables:**
- `epoch_3.pth` — trained Stage A checkpoint, `mIoU=23.61` (GaussianFormer3D's own real
  metric, evaluated on a 70-scene held-out subset).
- `G_0` cached for all 850 scenes (`/media/user/1TSSD/min/g0_cache/`).
- `scripts/check_vram_gs25600.py`, `scripts/generate_depth_gt_full.py`,
  `scripts/make_tiered_infos.py`, `scripts/make_frame0_infos.py` (this repo);
  `extract_g0_cache.py` (GaussianFormer3D repo).

**Exit checklist:**
- [x] `N_g=25,600` VRAM-confirmed comfortable before real training began.
- [x] Real, full-scale data pipeline wired and validated via a real training smoke test.
- [x] Small-tier gate check showed healthy, real held-out improvement before committing to
      the full run.
- [x] Full-scale Stage A trained; checkpoint and real mIoU recorded.
- [x] `G_0` cached for all 850 scenes, spot-checked for correct shape and correct
      scene-to-file mapping.

**Git tag:** `v0.5b-phase5b-full-dataset-stageA`

---

## Phase 6 — GF3D-faithful Stage B implementation — **NOT YET STARTED**

**Goal:** implement the `t>0` deformation module per the finalized design
(`docs/STAGE_B_GF3D_FAITHFUL_DESIGN.md`, Design B), reusing GaussianFormer3D's own real
`DeformableFeatureAggregation3D` and per-block sequence directly, applied to the reference
buffer's Gaussian state.

**Steps:**
1. Resolve Section 3.3's rotation-transform question (the one remaining open item blocking
   implementation) — a deliberate decision, not an assumption.
2. Implement the per-block iteration (Section 3.4) and the Design-B final update
   (`G_t = anchor^(L)`, Section 3.5) — reusing `DeformableFeatureAggregation3D`,
   `AnchorEncoder`, and this project's own existing `DeformHeadMu`/`DeformHeadR`
   (`deform_heads.py`, unchanged from every prior design).
3. Wire the reference buffer (`buffer.py`, unchanged) and `CurrentFrameEncoder` (unchanged)
   into the new deformation step.
4. Validate against `G_0`s already cached in Phase 5B — no need to re-run Stage A for any of
   this validation work.
5. Gate 1/2/3 style validation (matching this project's own established, cost-controlled
   testing discipline), using the mini-dataset pipeline (deliberately kept for exactly this
   purpose) before any full-scale Stage B training is attempted.

**Config used:** to be created, based on `docs/STAGE_B_GF3D_FAITHFUL_DESIGN.md`.

**Deliverables:** new deformation module implementation, config, and a real Gate 1/2/3 result
on the mini dataset before considering full-scale Stage B training.

**Exit checklist:**
- [ ] Section 3.3's rotation-transform question resolved and documented.
- [ ] Per-block iteration and Design-B final update implemented, matching the design doc's
      math exactly.
- [ ] Gate 1 (shapes, no crash, VRAM) passes on real data.
- [ ] Gate 2 (n=3, cheap signal check) run and assessed before any larger commitment.

**Git tag:** `v0.6-phase6-gf3d-faithful-implemented` (once complete).

---

## Phase 7 — Loss completion (`L_tv`, `L_lidar`)

**Goal:** add the remaining two loss terms, verify each independently.

**Status: needs revisiting once Phase 6's implementation exists.** The original plan and
config references below predate the GF3D-faithful pivot; the underlying loss concepts
(penalize jitter, penalize drift from LiDAR geometry) are likely still relevant, but exact
integration details need reconfirming against the new deformation module once it exists,
not assumed from the original plan.

**Steps (original plan, needs reconfirming):**
1. Implement `src/losses/tv_loss.py` per `design_doc_v2.md` §4 (penalizes **change** in
   `Δμ`/`Δr` across frames, not the motion itself).
2. Implement `src/losses/lidar_loss.py` (nearest-Gaussian or depth-consistency term against
   `P_t`). **Note:** Occ3D GT ships a separate `mask_lidar` array alongside `mask_camera`
   (`EXPERIMENT_LOG.md` 2026-07-14) — worth using as a candidate input here rather than
   inferring LiDAR visibility from geometry alone.
3. Re-run Stage 1 training with each loss ablated to zero individually, confirm the expected
   failure mode: no `L_tv` → visibly jittery Gaussian trajectories; no `L_lidar` → Gaussians
   drift away from LiDAR point geometry.
4. Settle on `λ_tv`, `λ_lidar` via a quick sweep — do this before any joint fine-tuning phase.

**Deliverables:** `src/losses/{tv_loss,lidar_loss}.py` (stub files already exist, not yet
implemented/verified — see `README.md`'s repository structure section), ablation-to-zero
visualizations, chosen `λ_tv`/`λ_lidar` values logged with the sweep results that justified
them.

**Exit checklist:**
- [ ] `L_tv`-ablated run visibly jitters more than the full-loss run (quantified)
- [ ] `L_lidar`-ablated run visibly drifts from LiDAR geometry more than the full-loss run
- [ ] Final `λ_tv`, `λ_lidar` chosen and recorded with justification in EXPERIMENT_LOG.md

**Git tag:** `v0.7-phase7-losses-complete`

---

## Phase 8 — Stage 2 joint fine-tuning

**Goal:** unfreeze Stage A, confirm joint fine-tuning improves over Stage-1-frozen.

**Status: needs revisiting once Phase 6's implementation exists.** The steps below are the
original plan; exact mechanics (which checkpoint, what LR ratio, whether an "unroll window"
concept applies the same way under the GF3D-faithful design) need reconfirming once that
implementation exists.

**Steps (original plan, needs reconfirming):**
1. Load Stage 1's best checkpoint.
2. Unfreeze Stage A; apply a reduced LR ratio.
3. Train per schedule; monitor for destabilization in the first few epochs.

**Deliverables:** trained Stage 2 (joint) checkpoint, training curves showing (in)stability
in early epochs.

**Exit checklist:**
- [ ] Stage 2 training completes without divergence
- [ ] Final Stage 2 checkpoint's per-frame IoU/mIoU on validation clips beats Stage 1's —
      **or**, if it doesn't, this is itself a logged finding (the frozen-vs-joint ablation
      answer), not a failure to fix at all costs

**Git tag:** `v0.8-phase8-stage2-joint`

---

## Phase 9 — Full evaluation

**Goal:** the actual numbers the report will include.

**Steps:**
1. Run both Stage 1 (frozen) and Stage 2 (joint) final checkpoints over held-out frames.
   Scope (mini-dataset vs. full-scale) to be decided based on remaining time budget once
   Phase 6/8 are complete.
2. Compute per-class and overall IoU/mIoU (matching the annotation protocol used for
   training — SurroundOcc for full-scale, Occ3D for mini).
3. Compute the temporal flicker metric (frame-to-frame voxel-label change rate at static
   regions).
4. Measure inference FPS/latency.
5. Baseline comparison: GaussianFormer3D run independently per frame (no temporal module at
   all) as the "no-memory" reference point.

**Deliverables:** final results table (IoU/mIoU per class + overall, flicker, FPS).

**Exit checklist:**
- [ ] All configurations evaluated on identical held-out frames
- [ ] Results table complete and saved
- [ ] At least one qualitative visualization saved for report figures

**Git tag:** `v0.9-phase9-evaluated`

---

## Phase 10 — Ablations

**Goal:** the ablation table for the report.

**Steps:** run each of the following, all evaluated identically to Phase 9:
1. Frozen-only vs. joint (Phase 8).
2. **Design A vs. Design B** (`docs/STAGE_B_GF3D_FAITHFUL_DESIGN.md`, Section 3.6) — Design B
   is the primary implementation; Design A is a documented, ready-to-run alternative, worth
   testing directly if Design B underperforms or as a completeness check either way.
3. `L_tv`/`L_lidar` ablations (Phase 7).
4. Section 3.3's rotation-transform decision (Phase 6) — with vs. without, if a real
   alternative exists once that question is resolved.

**Deliverables:** `experiments/ablations.md`.

**Exit checklist:**
- [ ] Every ablation row uses the identical eval protocol from Phase 9
- [ ] Table consolidated and ready to paste into the report's Experiments section

**Git tag:** `v0.10-phase10-ablations`

---

## Phase 11 — Report writing

**Goal:** submission-ready draft.

**Steps:** follow the outline in `docs/design_doc_v2.md`. Write Method and Experiments first,
grounded in `docs/STAGE_B_GF3D_FAITHFUL_DESIGN.md` for the architecture description.

**Exit checklist:**
- [ ] Every number in the Experiments/Ablations sections traces to a specific
      `experiments/*.md` file — no numbers written from memory
- [ ] Limitations section explicitly states: Stage A reused as a dependency rather than
      reimplemented (a legitimate design choice, stated plainly rather than implying original
      architecture work that didn't happen), Stage A trained to 3 of a planned 6 epochs given
      the project's overall time budget (not full convergence), and any scope limitation on
      the final evaluation (mini-dataset vs. full-scale, per Phase 9's decision)
- [ ] Draft reviewed with Prof. Chiang before submission

**Git tag:** `v1.0-submission`
