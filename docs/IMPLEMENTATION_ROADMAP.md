# Implementation Roadmap — Dynamic Gaussian Occupancy Prediction

Each phase has: **Goal**, **Steps**, **Config used**, **Deliverables**, **Exit checklist**,
**Git tag**. Do not start phase N+1 until phase N's exit checklist is fully checked — this
mirrors the discipline that worked in QG-Fusion (EXPERIMENT_LOG.md / ROADMAP.md source-of-truth
pattern). Log every run, pass or fail, in `EXPERIMENT_LOG.md` using the template in
`docs/EXPERIMENT_LOG_TEMPLATE.md`.

---

## Phase 0 — Environment, repo, and data verification

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

**Deliverables:** working env, `scripts/verify_scene_coverage.py`, resolved `requirements.txt`,
symlinked `data/`.

**Exit checklist:**
- [*] GaussianFormer3D repo runs its own demo end-to-end in the local environment
- [x] `requirements.txt` has concrete, tested version numbers (no `TBD` left)
- [x] All 10 mini scene names confirmed present in `data/occ3d/gts/`
- [x] `pc_range` / voxel size match verified in writing (paste output into EXPERIMENT_LOG.md)
- [x] Repo pushed to GitHub with this exit state tagged

Edited for First checklist:[*]
- GaussianFormer3D repo's core imports (mmdet3d/mmcv/spconv/DFA3D/LocalAggregator/
- GaussianOccEncoder3D) succeed cleanly; unittest_DFA3D.py passes. Full eval.py/train.py
- run against author-provided weights+data was NOT performed (deferred — not required
- since Occ4DGS writes its own Stage A training loop rather than reusing theirs as-is).

**Git tag:** `v0.0-phase0-env-verified`

---

## Phase 1 — Frame index & data loading

**Goal:** a reliable per-scene frame index that knows which frames have valid GT, before any
model touches the data.

**Steps:**
1. Build `src/datasets/nuscenes_mini.py`: loads the 10-scene nuscenes-devkit tables, returns
   ordered per-scene frame lists with sample tokens.
2. Build `src/datasets/occ3d_gt.py`: given a sample token, loads the Occ3D voxel label
   (reuse loader logic from the 3DGS project); returns `None`/flag if missing.
3. Tag every frame `has_gt: bool`; write out a JSON/pickle index
   `experiments/phase1_frame_index.json` — this is the single source of truth every later
   phase's dataloader reads from, so it isn't recomputed (and potentially reimplemented
   inconsistently) in Phase 4/5.
4. Slice contiguous valid-GT runs per scene; log the resulting run-length distribution
   (this tells you, per scene, how long a training clip can actually be before hitting a
   GT gap — directly relevant to whether a window of 2-3 frames is even always available).

**Config used:** `configs/dataset_mini_occ3d.yaml`.

**Deliverables:** `src/datasets/nuscenes_mini.py`, `src/datasets/occ3d_gt.py`,
`experiments/phase1_frame_index.json`, a short run-length histogram (paste into
EXPERIMENT_LOG.md, e.g. as a saved PNG or printed table).

**Exit checklist:**
- [ ] Frame index built for all 10 scenes, `has_gt` correctly reflects the known index-39 gap
- [ ] Every scene has at least one contiguous valid-GT run ≥ 3 frames (Stage 2's window
      length) — if any scene fails this, log it now and decide whether to exclude that scene
      or accept a shorter window for it
- [ ] Dataloader returns correctly-shaped camera tensors, LiDAR points, and GT voxels for a
      spot-checked sample from each of the 10 scenes (manually verify 10/10, not just 1)

**Git tag:** `v0.1-phase1-data-index`

---

## Phase 2 — Stage A standalone reproduction

**Goal:** GaussianFormer3D producing sane Gaussians on your machine, before Stage B exists.

**Steps:**
1. Implement/port Stage A per `docs/design_doc_v2.md` §1, using `configs/stage_a_gaussianformer3d.yaml`.
2. Run on **one** mini scene's frame 0 only. Log peak VRAM.
3. Visualize: scatter-plot of Gaussian positions (`open3d` or matplotlib 3D), histogram of
   scale/opacity values. Explicitly check for the failure modes you already debugged in
   QG-Fusion: position collapse, Z-axis collapse, scale saturation, uniform splatting.
4. If sane, run on all 10 scenes' frame 0; confirm no scene produces a degenerate result.
5. Optional: if time allows, train Stage A standalone for a few epochs against frame-0-only
   occupancy loss, to get a rough standalone IoU/mIoU number as a reference point (not the
   final metric, just a sanity baseline for later comparison).

**Config used:** `configs/stage_a_gaussianformer3d.yaml`.

**Deliverables:** working Stage A module (`src/models/stage_a_gaussianformer3d/`), scatter
plots for all 10 scenes' `G_0`, VRAM peak log.

**Exit checklist:**
- [ ] No position/Z-axis collapse, no scale saturation, on any of the 10 scenes' `G_0`
- [ ] Peak VRAM for Stage A alone recorded and comfortably under 24GB (headroom needed for
      Stage B's unroll later — flag now if Stage A alone is already tight)
- [ ] (Optional) standalone frame-0 IoU/mIoU logged as a reference number

**Git tag:** `v0.2-phase2-stageA-reproduced`

---

## Phase 3 — Stage C wiring smoke test

**Goal:** verify Gaussian-to-voxel splatting and the occupancy loss work end-to-end, using
Stage A's cached output, before Stage B introduces any new complexity.

**Steps:**
1. Implement `src/models/stage_c_splatting/` per GaussianFormer3D's own splatting formulation.
2. Feed a cached `G_0` (zero deformation) directly into splatting → `Ô`.
3. Implement `src/losses/occ_loss.py` (CE + Lovász-softmax vs Occ3D GT).
4. Confirm loss decreases over a few optimization steps on a single frame (overfit sanity
   check — this should work almost immediately since it's just Stage A's own reconstruction task).

**Config used:** `configs/stage_a_gaussianformer3d.yaml` (Stage A frozen/cached), no Stage B yet.

**Deliverables:** `src/models/stage_c_splatting/`, `src/losses/occ_loss.py`, an overfit-loss
curve screenshot/log for one frame.

**Exit checklist:**
- [ ] `Ô` shape matches Occ3D grid shape `[200, 200, 16]` with 18 classes
- [ ] Loss decreases monotonically (or near-monotonically) on the single-frame overfit test
- [ ] Gradients confirmed flowing back into Stage A's Gaussian parameters (check `.grad` is
      non-None and non-zero on a few sampled Gaussian params)

**Git tag:** `v0.3-phase3-stageC-smoketest`

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
   buffer.write(G_t) → Stage C splatting`, on a 2-frame toy sequence.
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
   frame encoder (§2.2 of `design_doc_v2.md` — reuse, don't reimplement).
2. Build `F^3D_t = F^d_t ⊗ F^c_t` for the current frame; feed pooled features to `hypernet`.
3. Train per `configs/stage_b_temporal.yaml: stage_1_warmup` (frozen Stage A, `L_occ` only
   first — hold off on `L_tv`/`L_lidar` until Phase 6 so a bug in either doesn't mask whether
   the core deformation is learning anything at all).
4. Start with `unroll_window: 2`, 1-2 scenes only; profile VRAM before scaling to all 10.
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
   `P_t`).
3. Re-run Stage 1 training with each loss ablated to zero individually, confirm the expected
   failure mode: no `L_tv` → visibly jittery Gaussian trajectories (plot a few Gaussian
   position tracks over the unrolled window); no `L_lidar` → Gaussians drift away from LiDAR
   point geometry (visualize overlay).
4. Settle on `λ_tv`, `λ_lidar` via the quick sweep `{0.01, 0.1, 1.0}` mentioned in the original
   design doc — do this now, on the small Stage-1 setup, before Stage 2's more expensive joint
   training.

**Config used:** `configs/stage_b_temporal.yaml: stage_1_warmup` + new loss weight fields.

**Deliverables:** `src/losses/{tv_loss,lidar_loss}.py`, ablation-to-zero visualizations,
chosen `λ_tv`/`λ_lidar` values logged with the sweep results that justified them.

**Exit checklist:**
- [ ] `L_tv`-ablated run visibly jitters more than the full-loss run (quantified, e.g. frame-
      to-frame position variance, not just "looks jittery")
- [ ] `L_lidar`-ablated run visibly drifts from LiDAR geometry more than the full-loss run
- [ ] Final `λ_tv`, `λ_lidar` chosen and recorded with justification in EXPERIMENT_LOG.md

**Git tag:** `v0.6-phase6-losses-complete`

---

## Phase 7 — Stage 2 joint fine-tuning

**Goal:** unfreeze Stage A, confirm joint fine-tuning improves over Stage-1-frozen.

**Steps:**
1. Load Stage 1's best checkpoint (`init_from` in `configs/stage_b_temporal.yaml`).
2. Unfreeze Stage A; apply the 10x-lower LR ratio from the config.
3. Increase `unroll_window` to 3; re-profile VRAM (this is the point where Phase 5's VRAM
   headroom check pays off — if it was tight at window=2 frozen, joint+window=3 will be
   tighter still; be ready to fall back to window=2 for Stage 2 if needed, and log that
   fallback explicitly rather than silently changing the config).
4. Train per schedule; monitor for destabilization in the first few epochs (large loss
   spikes) — if seen, this is the signal to reduce Stage A's LR further, not to abandon joint
   fine-tuning outright.

**Config used:** `configs/stage_b_temporal.yaml: stage_2_joint`.

**Deliverables:** trained Stage 2 (joint) checkpoint, VRAM profile at `window=3` joint,
training curves showing (in)stability in early epochs.

**Exit checklist:**
- [ ] Stage 2 training completes without divergence (loss curve stable after any initial
      transient)
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
4. Measure inference FPS/latency (no compression in this work, so this is a genuine speed
   story vs. re-running GaussianFormer3D from scratch every frame — measure that comparison
   explicitly).
5. Baseline comparison: GaussianFormer3D run independently per frame (no temporal module at
   all) as the "no-memory" reference point.

**Config used:** both `stage_1_warmup` and `stage_2_joint` final checkpoints, eval-only.

**Deliverables:** final results table (IoU/mIoU per class + overall, flicker, FPS) for: (a)
no-memory baseline, (b) Stage 1 frozen, (c) Stage 2 joint.

**Exit checklist:**
- [ ] All three configurations evaluated on identical held-out frames (no evaluation-set
      leakage or inconsistency between runs)
- [ ] Results table complete and saved (`experiments/phase8_results.md` or `.csv`)
- [ ] At least one qualitative visualization (occupancy prediction vs GT, a few frames) saved
      for paper figures

**Git tag:** `v0.8-phase8-evaluated`

---

## Phase 9 — Ablations

**Goal:** the ablation table for the paper.

**Steps:** run each of the following, all evaluated identically to Phase 8:
1. Frozen-only vs. joint (already have both from Phase 7/8 — just tabulate together).
2. `N_g = 12,800` vs. `6,400`, if VRAM allows after Phase 7/8's profiling.
3. `unroll_window = 2` vs. `3` at Stage 2 (if not already forced to 2 by VRAM in Phase 7).
4. `L_tv`/`L_lidar` ablations (already run in Phase 6 — re-confirm on full 10-scene eval,
   not just the small Phase 6 setup, since Phase 6 used a reduced scope).

**Deliverables:** `experiments/phase9_ablations.md` — one consolidated table.

**Exit checklist:**
- [ ] Every ablation row uses the identical eval protocol from Phase 8 (same held-out frames,
      same metrics)
- [ ] Table consolidated and ready to paste into the paper's Experiments section

**Git tag:** `v0.9-phase9-ablations`

---

## Phase 10 — Paper writing

**Goal:** submission-ready draft.

**Steps:** follow the outline in `docs/design_doc.md` §6 (Introduction, Related Work, Method,
Experiments, Ablations, Limitations, Conclusion). Write Method and Experiments first, since
those sections are now fully determined by Phases 0-9's actual results — don't draft
Introduction/Related Work first and then discover the Method section doesn't match what was
actually built.

**Exit checklist:**
- [ ] Every number in the Experiments/Ablations sections traces to a specific
      `experiments/phaseN_*.md` file — no numbers written from memory
- [ ] Limitations section explicitly states: no compensated-Gaussian/disocclusion handling,
      no compression (future work), 10-scene scope, `N_g` reduced from paper's default
- [ ] Draft reviewed with Prof. Chiang before submission

**Git tag:** `v1.0-submission`
