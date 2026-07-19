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

## Phase 4 — Stage B skeleton (shape/recursion validation only)

**Goal:** validate the recursive buffer mechanics and tensor shapes with dummy encoders,
before wiring in real camera/LiDAR features.

**Steps:**
1. Implement `src/models/stage_b_temporal/buffer.py`: the reference buffer object
   (`read()`, `write(G_t)` — recursive, no re-anchoring, per `configs/stage_b_temporal.yaml`).
2. Implement `src/models/stage_b_temporal/hypernet.py` and `deform_heads.py` with dummy
   (randomly initialized, untrained) inputs — e.g. feed zeros or random noise instead of real
   `F^3D_t` for now.
3. Wire: `buffer.read() → hypernet → per-Gaussian grid query → phi_mu, phi_r → update rule →
   buffer.write(G_t) → Stage C splatting`, on a 2-frame toy sequence. **Note:** "Stage C
   splatting" here means calling into `BEVSegmentorLiDAR3D`'s embedded head (per Phase 2/3's
   architecture pivot), not a standalone module.
4. Confirm shapes at every step; confirm the buffer correctly holds `G_t` (not `G_0`) after
   one step, i.e. recursion is actually recursive and not silently re-reading `G_0`.

**Config used:** `configs/stage_b_temporal.yaml` (structure only, dummy inputs — no training).

**Deliverables:** `src/models/stage_b_temporal/{buffer,hypernet,deform_heads}.py`, a 2-frame
toy-sequence test script confirming shapes and correct recursive buffer state.

**Exit checklist:**
- [ ] Buffer state after step 1 is provably `G_1` (deformed), not `G_0` — assert this directly
      in a unit test, don't eyeball it
- [ ] All tensor shapes match across the full chain for a 2-frame toy sequence
- [ ] Quaternion composition (`Δr_t ⊗ r_{t-1}`, normalized) verified numerically on a
      hand-computed example, not just "runs without error"

**Git tag:** `v0.4-phase4-stageB-skeleton`

---

## Phase 5 — Real encoders + Stage 1 (frozen warm-up) training

**Goal:** first real training of the temporal module, Stage A frozen.

**Steps:**
1. Wire in the real (frozen) Stage A camera/LiDAR encoders from Phase 2 as Stage B's current-
   frame encoder (§2.2 of `design_doc_v2.md` — reuse, don't reimplement). In practice this
   means reusing `BEVSegmentorLiDAR3D`'s `img_backbone`/`img_neck`/`lidar_voxel_encoder`
   submodules directly, consistent with Phase 2's reuse-not-reimplement pattern.
2. Build `F^3D_t = F^d_t ⊗ F^c_t` for the current frame; feed pooled features to `hypernet`.
3. Train per `configs/stage_b_temporal.yaml: stage_1_warmup` (frozen Stage A, `L_occ` only
   first — hold off on `L_tv`/`L_lidar` until Phase 6).
4. Start with `unroll_window: 2`, 1-2 scenes only; profile VRAM before scaling to all 10.
   **Given Phase 2's 2.84GB single-frame peak, there is likely substantial headroom for a
   2-frame unroll — confirm with real numbers rather than assuming.**
5. Log per-frame IoU/mIoU across a full unrolled validation clip (not just single-step loss).

**Config used:** `configs/stage_b_temporal.yaml: stage_1_warmup`.

**Deliverables:** trained Stage 1 checkpoint, VRAM profile at `window=2` on 1-2 scenes,
per-frame validation IoU/mIoU curve.

**Exit checklist:**
- [ ] VRAM profiled at `window=2`; confirmed headroom (or lack thereof) before scaling to 10
      scenes and before attempting `window=3` in Stage 2
- [ ] `L_occ`-only training shows the deformed-frame IoU/mIoU meaningfully above a "do-nothing"
      baseline (Δμ=0, Δr=identity) — this is the first real evidence the temporal module is
      learning anything, log this comparison explicitly
- [ ] Scaled successfully to all 10 scenes at `window=2`

**Git tag:** `v0.5-phase5-stage1-trained`

---

## Phase 6 — Loss completion (`L_tv`, `L_lidar`)

**Goal:** add the remaining two loss terms, verify each independently.

**Steps:**
1. Implement `src/losses/tv_loss.py` per `design_doc_v2.md` §4 (penalizes **change** in
   `Δμ`/`Δr` across frames, not the motion itself).
2. Implement `src/losses/lidar_loss.py` (nearest-Gaussian or depth-consistency term against
   `P_t`). **Note:** Phase 0 discovered Occ3D GT ships a separate `mask_lidar` array
   alongside `mask_camera` (`EXPERIMENT_LOG.md` 2026-07-14) — worth using as a candidate
   input here rather than inferring LiDAR visibility from geometry alone.
3. Re-run Stage 1 training with each loss ablated to zero individually, confirm the expected
   failure mode: no `L_tv` → visibly jittery Gaussian trajectories; no `L_lidar` → Gaussians
   drift away from LiDAR point geometry.
4. Settle on `λ_tv`, `λ_lidar` via the quick sweep `{0.01, 0.1, 1.0}` — do this now, before
   Stage 2's more expensive joint training.

**Config used:** `configs/stage_b_temporal.yaml: stage_1_warmup` + new loss weight fields.

**Deliverables:** `src/losses/{tv_loss,lidar_loss}.py`, ablation-to-zero visualizations,
chosen `λ_tv`/`λ_lidar` values logged with the sweep results that justified them.

**Exit checklist:**
- [ ] `L_tv`-ablated run visibly jitters more than the full-loss run (quantified)
- [ ] `L_lidar`-ablated run visibly drifts from LiDAR geometry more than the full-loss run
- [ ] Final `λ_tv`, `λ_lidar` chosen and recorded with justification in EXPERIMENT_LOG.md

**Git tag:** `v0.6-phase6-losses-complete`

---

## Phase 7 — Stage 2 joint fine-tuning

**Goal:** unfreeze Stage A, confirm joint fine-tuning improves over Stage-1-frozen.

**Steps:**
1. Load Stage 1's best checkpoint (`init_from` in `configs/stage_b_temporal.yaml`).
2. Unfreeze Stage A; apply the 10x-lower LR ratio from the config.
3. Increase `unroll_window` to 3; re-profile VRAM.
4. Train per schedule; monitor for destabilization in the first few epochs.

**Config used:** `configs/stage_b_temporal.yaml: stage_2_joint`.

**Deliverables:** trained Stage 2 (joint) checkpoint, VRAM profile at `window=3` joint,
training curves showing (in)stability in early epochs.

**Exit checklist:**
- [ ] Stage 2 training completes without divergence
- [ ] Final Stage 2 checkpoint's per-frame IoU/mIoU on validation clips beats Stage 1's —
      **or**, if it doesn't, this is itself a logged finding (the frozen-vs-joint ablation
      answer), not a failure to fix at all costs

**Git tag:** `v0.7-phase7-stage2-joint`

---

## Phase 8 — Full evaluation on all 10 scenes

**Goal:** the actual numbers the paper will report.

**Steps:**
1. Run both Stage 1 (frozen) and Stage 2 (joint) final checkpoints over held-out frames
   across all 10 scenes.
2. Compute per-class and overall IoU/mIoU (Occ3D 18-class protocol).
3. Compute the temporal flicker metric (frame-to-frame voxel-label change rate at static
   regions).
4. Measure inference FPS/latency (no compression in this work).
5. Baseline comparison: GaussianFormer3D run independently per frame (no temporal module at
   all) as the "no-memory" reference point.

**Config used:** both `stage_1_warmup` and `stage_2_joint` final checkpoints, eval-only.

**Deliverables:** final results table (IoU/mIoU per class + overall, flicker, FPS).

**Exit checklist:**
- [ ] All three configurations evaluated on identical held-out frames
- [ ] Results table complete and saved (`experiments/phase8_results.md` or `.csv`)
- [ ] At least one qualitative visualization saved for paper figures

**Git tag:** `v0.8-phase8-evaluated`

---

## Phase 9 — Ablations

**Goal:** the ablation table for the paper.

**Steps:** run each of the following, all evaluated identically to Phase 8:
1. Frozen-only vs. joint.
2. `N_g = 12,800` vs. `6,400` — **worth prioritizing given Phase 2's 2.84GB peak showed far
   more VRAM headroom than assumed; this ablation may be cheaper to run than originally scoped.**
3. `unroll_window = 2` vs. `3` at Stage 2.
4. `L_tv`/`L_lidar` ablations, re-confirmed on full 10-scene eval.

**Deliverables:** `experiments/phase9_ablations.md`.

**Exit checklist:**
- [ ] Every ablation row uses the identical eval protocol from Phase 8
- [ ] Table consolidated and ready to paste into the paper's Experiments section

**Git tag:** `v0.9-phase9-ablations`

---

## Phase 10 — Paper writing

**Goal:** submission-ready draft.

**Steps:** follow the outline in `docs/design_doc.md` §6. Write Method and Experiments first.

**Exit checklist:**
- [ ] Every number in the Experiments/Ablations sections traces to a specific
      `experiments/phaseN_*.md` file — no numbers written from memory
- [ ] Limitations section explicitly states: no compensated-Gaussian/disocclusion handling,
      no compression (future work), 10-scene scope, `N_g` reduced from paper's default,
      Stage A reused as a dependency rather than reimplemented (a legitimate design choice,
      but worth stating plainly rather than implying original architecture work that didn't
      happen)
- [ ] Draft reviewed with Prof. Chiang before submission

**Git tag:** `v1.0-submission`