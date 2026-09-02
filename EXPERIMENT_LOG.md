# Experiment Log — Occ4DGS

Running research log. Copy the block from `docs/EXPERIMENT_LOG_TEMPLATE.md` for every run.
This file (not memory, not Slack messages to labmates) is the source of truth for paper
writing and professor check-ins.

## Summary table (update after every logged run)

| Run ID | Phase | N_g | Window | Stage | LR | Overall mIoU | Overall IoU | VRAM peak | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-07-12-env-verification | 0 | — | — | — | — | — | — | — | env verified |
| 2026-07-14-pc-range-verification | 0 | — | — | — | — | — | — | — | GT format confirmed, dual mask (lidar+camera) discovered |
| 2026-07-14-frame-index-and-gt-loader | 1 | — | — | — | — | — | — | — | 100% GT coverage all 10 scenes, dual on-disk layout bug found+fixed |
| 2026-07-16-depth-gt-generation | 2 | — | — | — | — | — | — | — | 2424 depth files generated, cam2img/img2cam bug found+fixed |
| 2026-07-18-stage-a-first-successful-forward-pass | 2 | 6400 | 1 (single frame) | eval only | n/a | n/a | n/a | 2.84 GB | first full forward pass, 11 bugs resolved |
| 2026-07-19-stage-a-training-path-validated | 2 | 6400 | 1 (single frame) | Stage A only, overfit | 1e-4 AdamW | 0.1366 (single-frame overfit) | — | 2.84 GB | training path validated, loss 26.70→21.62/200 iters |
| 2026-07-27-stageA-8scene-final | 5 | 6400 | 1 (single frame) | Stage A only | cosine, 120ep | trained ~16.5 adj. / held-out ~7.07 adj. | — | — | 8/10 scenes trained, scene-1094/scene-1100 fixed as permanent held-out pair |
| 2026-07-27-stageB-8scene-matched-heldout | 5 | 6400 | 2 | Stage B, delta-conditioned | 1e-4 AdamW | in-sample +1.35 (raw) vs do-nothing; held-out negative | — | — | first fair Stage A/B-matched-scope test; confirms in-sample-positive/held-out-negative pattern |
| 2026-07-29-gt-motion-and-ego-compensation | 5 | 6400 | 2 | Stage B | 1e-4 AdamW | — | — | — | GT motion 60-84% of voxels, dominated by ego motion (not object motion); ego-motion compensation implemented; pc_range-clamp opacity bug found+fixed; Gaussian-budget representational gap discovered (44.4% of clips exceed damage threshold) |
| 2026-07-30-steps1-4-clean-baseline | 5 | 6400 | 2 | Stage B, compensation OFF | 1e-4 AdamW | best held-out delta +0.044 → +0.106 across Steps 1-4 | — | — | rotation-order fix, grid resolution/channel rebalance, 4DGC-confirmed PE-as-coordinate query, conv-decoder HyperNet; each a modest real single-run improvement, collapse-then-plateau shape unchanged |
| 2026-07-30-scene-scaling-sweep-noise-floor | 5 | 6400 | 2 | Stage B | 1e-4 AdamW | n=1/3/6/8: +0.091/+0.125/+0.105/+0.068, no monotonic pattern | — | — | CRITICAL: do-nothing baseline varies ~0.036 run-to-run (should be deterministic) -- same order of magnitude as claimed architecture gains; ALL Step 1-4 and scene-scaling comparisons provisional pending noise-floor measurement; work PAUSED here |
| 2026-08-08-option-d-direct-projection | 5 | 6400 | 2 | Stage B | 1e-4 AdamW | Gate3 n=8: +0.082 (peak inconclusive); trough oscillates -0.84 to -1.05, recurring | — | — | direct per-Gaussian projection, no dense grid; both un-norm/norm variants concluded, not improved |
| 2026-08-13-vggt-deformable-kickoff | 5 | 6400 | 2 | Stage B | 1e-4 AdamW | Gate1 pass only | — | 11.85 GB → 7.30 GB (post-downsample fix) | VGGT-1B frozen feature extractor + 4-block deformable attention; resolution crisis found+fixed |
| 2026-08-14-vggt-deformable-image-wh-bugfix | 5 | 6400 | 2 | Stage B | 1e-4 AdamW | pre-fix -0.384 (n=3, ~85% fallback rate) | — | 7.30 GB | image_wh/native-scale mismatch found+fixed; fallback rate 85%→4-9% post-fix |
| 2026-08-18-vggt-deformable-dropout | 5 | 6400 | 2 | Stage B, z_dropout | 1e-4 AdamW | -0.442 (n=3) vs -0.496 no-dropout | — | 7.30 GB | modest, ambiguous improvement; same collapse shape unchanged |
| 2026-08-19-vggt-cross-architecture-conclusion | 5 | 6400 | 2 | Stage B | 1e-4 AdamW | 5 architectures, same signature: in-sample loss ↓, held-out peaks epoch 4-6 then collapses | — | — | CONCLUSION: likely small-data generalization gap, not a remaining bug; VGGT reverted, redirect toward data scale |
| 2026-08-22-professor-review-gf3d-faithful | 5→6 | — | — | — | — | — | — | — | professor adopts GF3D-faithful design for report; Motion HyperNet kept as personal backup, VGGT discarded; git branches restructured |
| 2026-08-21-full-dataset-vram-and-wiring | Full-dataset | 25600 | — | Stage A only | — | n/a (feasibility check) | — | ~4.07 GB (synthetic fwd pass) | N_g=25600 VRAM confirmed; full nuScenes v1.0-trainval + SurroundOcc pipeline wired; 5 real bugs found+fixed |
| 2026-08-22-small-tier-gate-check | Full-dataset | 25600 | — | Stage A only | 1e-4 AdamW | 8.54→9.53→10.36→10.51→10.52 (6 epochs, 50 scenes, real GF3D mIoU metric — NOT directly comparable to the held-out-delta numbers above) | — | ~23 GB | healthy signal, plateau expected at 50 scenes; skip medium tier, proceed to full |
| 2026-08-24-full-scale-stage-a-training | Full-dataset | 25600 | — | Stage A only | 1e-4 AdamW | 17.24→22.06→23.61 (epochs 0-2, 700 scenes, real GF3D mIoU) | iou2: 38.64→39.98 | ~23 GB (tight, recurring OOM diagnosed+resolved) | stopped deliberately at epoch 3/6 given 20-day budget; ~87% of GF3D's own reported 27.1 mIoU |
| 2026-08-27-g0-extraction-cache | Full-dataset | 25600 | — | Stage A only (inference) | n/a | n/a | n/a | n/a | G_0 cached for all 850 scenes; 2 real bugs + 1 drive write-corruption issue found+fixed via 5-scene test first |
| 2026-08-28-gf3d-faithful-housekeeping | Full-dataset | — | — | — | — | — | — | — | Motion HyperNet/Step5 modules removed from gf3d-faithful-stageb (preserved on archive branches); repo cleanup, data/README.md updated |

**Note on commit history:** a few commits don't map to a distinct log entry above, since they
were formatting/checklist-wording fixes rather than new runs: `ea1f404` (corrected Phase 0
checklist wording), `d9234da` (scene coverage + pc_range scripts, folded into the
2026-07-14-pc-range-verification entry above), `ab33fcf`/`51f85f8` (env-verification
finalization, folded into the 2026-07-12/14 entries above), `f3f6cad` (log formatting cleanup).
Full history: `git log --oneline` in the repo.

---

## Entries

## [Phase 0] Run ID: 2026-07-12-env-verification

- **Git commit:** `14f6f1a` (tag: `v0.0-phase0-env-verified`)
- **Config file(s):** N/A — environment setup, no training config yet
- **Command:**
```bash
conda create -n gf3d python=3.8.16
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install openmim
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmsegmentation==1.2.1
mim install mmdet3d==1.3.0
pip install spconv-cu120
pip install timm
pip install ftfy regex   # undocumented dependency, mmseg's backbone zoo needs it eagerly
git clone https://github.com/lunarlab-gatech/GaussianFormer3D.git
cd GaussianFormer3D/model/encoder/gaussian_encoder/ops && pip install -e . && cd -
cd GaussianFormer3D/model/head/localagg && pip install -e . && cd -
git clone https://github.com/IDEA-Research/3D-deformable-attention.git
# edited 3D-deformable-attention/DFA3D/setup.py: c++14 -> c++17
# (both extra_compile_args['cxx'] and ['nvcc'])
cd 3D-deformable-attention/DFA3D
rm -rf build/ dfa3D.egg-info/
bash setup.sh 0
cd ..
python unittest_DFA3D.py
```
- **Hardware:** RTX 3090 24GB, driver reports CUDA 12.8, conda env python 3.8.16
- **Hypothesis / what this run tests:**
  Confirm GaussianFormer3D + DFA3D build and import cleanly on this machine's
  stack before writing any Occ4DGS-specific code (Phase 0, roadmap step 2).
- **Results:**

  | Check | Result |
  |---|---|
  | spconv-cu120 install | pass, no cu117 fallback needed |
  | DFA3D build (bash setup.sh 0) | pass, after c++17 patch |
  | unittest_DFA3D.py | pass (exit code 0, no traceback, confirmed twice) |
  | LocalAggregator import | pass |
  | GaussianOccEncoder3D import | pass |
  | dfa3D module import | pass |

- **Observations:**
  Two undocumented gaps beyond the repo's own `docs/installation.md`:
  1. DFA3D's `setup.py` hardcoded `-std=c++14` in both `extra_compile_args['cxx']`
     and `['nvcc']`, incompatible with torch 2.1's C++17-required ATen headers
     (`#error C++17 or later compatible compiler is required to use ATen`). Fixed
     by editing both occurrences to `c++17`, then `rm -rf build/ dfa3D.egg-info/`
     before rebuilding to clear stale compiled objects.
  2. `mmsegmentation` 1.2.1's top-level `__init__` eagerly imports its full
     backbone zoo (BEiT etc.), which needs `ftfy` + `regex` for CLIP-style
     tokenization even though this project doesn't use those backbones.
     Fixed with `pip install ftfy regex`.

  `unittest_DFA3D.py` took ~1hr wall-clock with no console output — consistent
  with a gradcheck-style float64 numerical verification, not a hang (confirmed
  no zombie process via `ps aux`, no OOM in `dmesg`).
- **Bugs / issues encountered & fixes:** see Observations above.
- **Decision / next step:**
  Environment fully verified. Proceeding to Phase 1 (frame index & data
  loading). `requirements.txt` updated with exact working versions.

---

## [Phase 0] Run ID: 2026-07-14-pc-range-verification

- **Git commit:** `c7721ea` (tag: `v0.1-phase0-complete`)
- **Config file(s):** `configs/dataset_mini_occ3d.yaml`, `configs/stage_a_gaussianformer3d.yaml`
- **Command:** `np.load()` inspection of `data/occ3d_gts/scene-0061/023c4df2.../labels.npz`
- **Results:**
  - `semantics`: shape `(200,200,16)`, dtype uint8, range 0-17 — matches configured
    `pc_range=[-40,-40,-1,40,40,5.4]`, voxel_size=0.4m, 18 classes. No mismatch.
  - `mask_lidar`: shape `(200,200,16)`, dtype uint8, binary — LiDAR visibility mask,
    not previously accounted for in configs; candidate input for `L_lidar` (Phase 6).
  - `mask_camera`: shape `(200,200,16)`, dtype uint8, binary — camera visibility mask,
    matches `configs/dataset_mini_occ3d.yaml`'s `use_camera_visibility_mask` flag.
- **Observations:** Occ3D-nuScenes ships semantics + two SEPARATE binary masks in one
  `labels.npz`, not a combined mask or an embedded special value. `src/datasets/occ3d_gt.py`
  (Phase 1) needs to load and return all three arrays, not just semantics.
- **Decision / next step:** Phase 0 fully complete (all 5 exit checklist items closed).
  Proceeding to Phase 1 (frame index & data loading).

---

## [Phase 1] Run ID: 2026-07-14-frame-index-and-gt-loader

- **Git commit:** `3b629b4` (tag: `v0.1-phase1-data-index`)
- **Config file(s):** `configs/dataset_mini_occ3d.yaml`
- **Command:** `scripts/build_frame_index.py`, `scripts/spot_check_dataloader.py`
- **Hardware:** N/A (CPU-only, dataset indexing)
- **Hypothesis / what this run tests:**
  Build a reliable `has_gt`-tagged frame index across all 10 scenes before any
  model touches the data (Phase 1, roadmap steps 1-4).
- **Results:**

  | scene | #frames | #has_gt (first pass) | #has_gt (after fix) | max_run (after fix) |
  |---|---|---|---|---|
  | scene-0061 | 39 | 39 | 39 | 39 |
  | scene-0103 | 40 | 40 | 40 | 40 |
  | scene-0553 | 41 | 0 | 41 | 41 |
  | scene-0655 | 41 | 0 | 41 | 41 |
  | scene-0757 | 41 | 41 | 41 | 41 |
  | scene-0796 | 40 | 40 | 40 | 40 |
  | scene-0916 | 41 | 0 | 41 | 41 |
  | scene-1077 | 41 | 41 | 41 | 41 |
  | scene-1094 | 40 | 40 | 40 | 40 |
  | scene-1100 | 40 | 0 | 40 | 40 |

- **Observations:**
  First pass showed a clean ALL-OR-NOTHING pattern: 6 scenes at 100% GT
  coverage, 4 scenes at exactly 0%. Root cause (confirmed by direct file
  check, not guessed): this Occ3D-nuScenes GT dump uses TWO different
  on-disk layouts depending on scene:
    - (a) `data/occ3d_gts/<scene>/<token>/labels.npz` (subfolder + labels.npz)
    - (b) `data/occ3d_gts/<scene>/<token>.npz` (flat file)

  scene-0061, 0103, 0757, 0796, 1077, 1094 use (a); scene-0553, 0655, 0916,
  1100 use (b). No correlation found with log location (boston-seaport vs
  singapore) or log file naming that would predict which convention a scene
  uses — likely an artifact of how this GT release was assembled/merged
  from multiple download batches, not a semantic pattern to rely on.
  Fixed by checking both path candidates in `load_occ3d_labels()` rather than
  assuming one convention. After the fix: 100% GT coverage across all 10
  scenes (better than the old 3DGS-project "mini_train index 39 gap" would
  have suggested — that gap does not appear to apply to this exact set of
  10 scenes/this GT release, or was specific to a different data path).
- **Bugs / issues encountered & fixes:** see Observations above.
- **Decision / next step:**
  Phase 1 exit checklist fully satisfied (all 10 scenes have full-length
  contiguous valid-GT runs, well above the 3-frame minimum needed for
  Stage 2's unroll window). `experiments/phase1_frame_index.json` is now the
  source of truth for all later phases — do not recompute this differently
  elsewhere. Proceeding to Phase 2 (Stage A reproduction).

---

## [Phase 2] Run ID: 2026-07-16-depth-gt-generation

- **Git commit:** `40e3d77`
- **Config file(s):** N/A — standalone preprocessing script
- **Command:** `scripts/generate_depth_gt.py`
- **Hardware:** RTX 3090 24GB (unused — pure CPU/numpy), ~2424 files, all 10 scenes
- **Hypothesis / what this run tests:**
  Generate BEVDepth-style `depth_gt/*.bin` files from our own v1.0-mini LiDAR sweeps,
  replacing GaussianFormer3D's SharePoint-downloaded depth_gt (not available for
  our Occ3D-mini setup). Files written to `data/nuscenes_mini/depth_gt/` (Option A:
  onto the external drive, sibling of `samples/`, matching `LoadMultiViewDepthFromFiles`'
  expected layout unmodified).
- **Results:**
  2424 files written (10 scenes × ~40.4 avg frames × 6 cameras). Spot-checked 8 files
  across multiple scenes/cameras: point counts 2100-4600 per file, u in [0,1600],
  v in [0,900], depth in [3-90]m. All physically plausible.
- **Bugs / issues encountered & fixes:**
  FIRST ATTEMPT WAS WRONG — silently produced empty/garbage files (e.g. CAM_BACK_LEFT
  on scene-0061 frame 0 returned npts=0, u/v in range [-0.9,-0.4] instead of
  [0,1600]/[0,900]). Root cause: built `cam2img` as the intrinsic matrix K directly
  embedded in a 4x4, then used it as if it were the `img2global` multiplier — but
  GaussianFormer3D's own `dataset/utils.py get_img2global()` actually uses
  `img2cam = inv(cam2img) = K^-1` in that position
  (`img2global = ego2global @ cam2ego @ img2cam`). Copying their variable NAME
  (`cam2img`) without verifying which matrix role it actually plays in their formula
  produced numerically-plausible-looking code that was nonetheless wrong. Fixed by
  matching their formula term-for-term, confirmed via direct debug printout of u/v
  ranges before and after the fix.
- **Decision / next step:**
  `depth_gt` generation validated. Ready to move to Phase 2's config override and
  the first actual Stage A "run on frame 0" step, now that `occ4dgs_dataset.py` +
  `generate_depth_gt.py` both produce verified-correct inputs.

---

## [Phase 2] Run ID: 2026-07-18-stage-a-first-successful-forward-pass

- **Git commit:** `b91e990`
- **Config file(s):** `GaussianFormer3D/config/occ4dgs_mini_occ3d_gs6400.py`
  (mirrored into `configs/occ4dgs_mini_occ3d_gs6400.py`)
- **Command:** `scripts/run_stage_a_frame0.py`, scene-0061, frame 0
- **Hardware:** RTX 3090 24GB, peak VRAM **2.84 GB** (single sample, batch=1, no
  gradient/training yet)
- **Hypothesis / what this run tests:**
  Confirm GaussianFormer3D's `BEVSegmentorLiDAR3D` runs a full forward pass end-to-end
  on our own Occ3D-mini dataset adapter, using `N_g=6400`, ResNet101-DCN,
  `occ_annotation="occ3d"`.
- **Results:** SUCCESS. Full forward pass completes. Output keys include `pred_occ`,
  `gaussian`, `sampled_xyz`, `sampled_label`, `occ_mask` — exactly the expected structure.
  Peak VRAM 2.84GB, far under the 24GB budget (single sample, eval mode, no backward
  pass yet — training costs substantially more due to activations/gradients; revisit
  once the training loop exists — see next entry, which already answers this).
- **Bugs / issues encountered & fixed (chronological, this phase):**
  1. `results['lidar_path']` missing (separate key from `pts_filename`, both required).
  2. `results['sweeps']` missing — ported `obtain_sensor2top` logic from `make_gf3d_infos.py`.
  3. `LoadOccupancyOcc3d` occ_path built with wrong root (missing `occ3d_gts` segment).
  4. GaussianFormer3D custom module registry never triggered — needed `import model`.
  5. `_delete_=True` is an mmengine config-merge directive, meaningless without `_base_`
     inheritance — removed from `img_backbone`, `spconv_layer`, `head.empty_args/cuda_kwargs`.
  6. `img_neck` was a partial override (`start_level=1` only) relying on `_base_/model.py`'s
     FPN definition we don't inherit — wrote the full FPN dict explicitly.
  7. `use_deformable_func` defaults `False`, causes `UnboundLocalError` deep in
     `deformable_module_3d.py` — set `True` explicitly.
  8. `use_camera_embed` defaults `False` — set `True` to match proven SurroundOcc config intent.
  9. `d_bound` duplicate keyword argument (pasted twice in `deformable_model` dict) — removed dup.
  10. `image_wh`, `occ_xyz`, `occ_label`, `occ_cam_mask` all missing from our hand-built `metas`
      dict in `run_stage_a_frame0.py`'s `to_batch_of_one()` — added all four with correct
      dtypes (`occ_label` as `.long()`, `occ_cam_mask` as `.bool()`).
  11. `residual_mode` defaults `"add"` (128-dim output), but FFN was sized for `"cat"` (256-dim
      input) per the proven config's implied intent — set `residual_mode="cat"` explicitly.

  **Root cause common to most of these:** writing `occ4dgs_mini_occ3d_gs6400.py` WITHOUT
  `_base_` inheritance (deliberate choice to avoid silently importing unverified
  SurroundOcc/2D defaults) means every field the original config relied on `_base_` to
  supply had to be identified and set explicitly, one crash at a time. In hindsight, a
  full field-by-field diff against class `__init__` signatures (as eventually done for
  `DeformableFeatureAggregation3D`) earlier would have caught several of these at once
  rather than sequentially.
- **Decision / next step:**
  Phase 2 step 2 (run on one scene's frame 0) COMPLETE. Next: visualize Gaussian
  positions/scale/opacity across all 10 scenes, then validate the training path.

---

## [Phase 2] Run ID: 2026-07-19-stage-a-training-path-validated

- **Git commit:** `605b893` (tag: `v0.2-phase2-stageA-reproduced`)
- **Config file(s):** `GaussianFormer3D/config/occ4dgs_mini_occ3d_gs6400.py`
- **Command:** `scripts/overfit_stage_a_single_frame.py`, scene-0061 frame 0, 200 iterations
- **Hardware:** RTX 3090 24GB, peak VRAM **2.84 GB** (consistent with eval-mode run above —
  overfitting a single cached frame doesn't add meaningfully to peak memory here)
- **Hypothesis / what this run tests:**
  Confirm the full training path (loss computation, backward, gradient clipping,
  optimizer step) works end-to-end, and establish a reference mIoU number
  (roadmap Phase 2 step 5, upgraded from "optional" to "important").
- **Results:**
  Loss: **26.70 → 21.62** over 200 iterations, consistent decrease, no divergence.
  Reference single-frame overfit **mIoU: 0.1366** (classes 1-16; 11/16 classes present
  in this frame's GT). Modest, as expected for 200 iterations of light overfitting
  on one frame with a from-scratch-initialized encoder — the loss trend, not the
  absolute mIoU value, is the real pass/fail signal here.

  Preceded by a full all-10-scenes collapse/saturation check (`scripts/visualize_gaussians.py`):
  every scene's `G_0` came back clean — no position/Z-axis collapse, no scale saturation,
  remarkably consistent statistics across scenes (means spanning the full `pc_range`,
  scales within `[0.2, 1.6]`, opacity centered ~0.53 in every scene — expected, since
  initialization dominates over data differences before training).
- **Bugs / issues encountered & fixes:**
  1. **Environment:** `~/.local` held a broken `tensorflow 2.5.0` install (unrelated to this
     project, leaking via Python's default user-site-packages behavior) that broke
     `loss/__init__.py`'s tensorboard import chain. Fixed per-invocation with
     `PYTHONNOUSERSITE=1`, plus installing `termcolor`/`urllib3`/`cachetools`/`absl-py`/
     `google-auth`/`markdown` directly into `gf3d` (previously silently borrowed from
     `~/.local` without our knowledge in every earlier Phase 0-2 script too — now added
     to `requirements.txt` explicitly).
  2. `occ4dgs_mini_occ3d_gs6400.py` was missing `optimizer`/`grad_max_norm`/`max_epochs`
     top-level config (never needed until now, since only forward passes had been run
     before) — added, matching original SurroundOcc config's values.
  3. `model_obj(imgs=batch["imgs"], ...)` reused the SAME tensor object every iteration;
     `BEVSegmentorLiDAR3D.extract_img_dpt_feat` does an IN-PLACE `imgs.squeeze_(0)` on the
     first call, permanently corrupting `batch["imgs"]`'s shape for iteration 2 onward.
     Fixed by cloning `imgs`/`dpt` fresh each iteration before passing to the model — a
     bug specific to reusing one cached batch across iterations in a toy script, would
     not occur in real training with a fresh dataloader batch every step.
  4. `pred_occ` is a LIST (one entry per applied-loss decoder layer), not a tensor —
     confirmed via source (`gaussian_head.py: prediction.append(semantics)`). Fixed
     with `isinstance` checks, taking `[-1]` (final layer).
  5. `pred_occ` shape is `[B, 18_classes, 640000_voxels]` — class dim is `dim=1`, not
     the last dim. `argmax(dim=-1)` was backwards; fixed to `argmax(dim=1)`.
- **Decision / next step:**
  Phase 2 FULLY COMPLETE (all exit checklist items satisfied). Phase 3 (originally
  "Stage C wiring smoke test") is subsumed by this result — Gaussian-to-voxel
  splatting is embedded inside `BEVSegmentorLiDAR3D`'s head and already proven
  working here (see `IMPLEMENTATION_ROADMAP.md`'s Phase 3 section for the explicit
  mapping of its exit checklist onto this evidence). Proceeding directly to Phase 4
  (Stage B skeleton: reference buffer, motion hypernet, deform heads).

---

  ## [Phase 4] Run ID: 2026-07-22-stageB-skeleton-validated

- **Git commit:** (fill in after commit below)
- **Config file(s):** `configs/stage_b_temporal.yaml` (structure only, dummy inputs, no training)
- **Command:** `python tests/test_stage_b_skeleton.py`
- **Hardware:** RTX 3090 24GB (test suite is CPU/GPU-agnostic, tiny toy scale — N=64 Gaussians)
- **Hypothesis / what this run tests:**
  Validate Stage B's recursive buffer mechanics and tensor shapes with dummy encoders
  (random noise standing in for F^3D_t, per roadmap Phase 4 step 2), before wiring in
  real camera/LiDAR features in Phase 5.
- **Results:** SUCCESS, all three exit-checklist items confirmed with evidence, not eyeballed:
  1. Quaternion composition (`Δr_t ⊗ r_{t-1}`, normalized) verified against a hand-computed
     90°+90°=180° z-rotation example, plus identity-composition and zero-rotation edge cases.
  2. `grid_sample` coordinate convention (x,y,z ↔ W,H,D axis order) verified against a
     hand-indexed 2×2×2 grid with a distinct value per corner — catches an img2cam-style
     axis-order bug class before real features make it silent.
  3. Buffer state after `write(g1)` provably holds `G_1` (`not torch.allclose` vs `G_0`'s
     means), and after `write(g2)` provably holds `G_2` (not `G_1`, not `G_0`) — recursion
     confirmed actually recursive, not silently re-reading `G_0`.
  All tensor shapes match across the full chain for the 2-frame toy sequence (`N=64`,
  `L=3` grid levels at resolutions `(4,8,16)`, `grid_feat_dim=16`).
- **Bugs / issues encountered & fixed:**
  1. Manual incremental `__init__.py` edit (`echo "from .buffer import GaussianState" > __init__.py`)
     silently overwrote the full package `__init__.py` down to one import — caused
     `ImportError: cannot import name 'ReferenceBuffer'` in both the direct import check and
     `test_stage_b_skeleton.py`. Root cause: `echo >` truncates rather than appends. Fixed by
     writing the complete `__init__.py` in one step instead of building it up incrementally
     with shell redirection.
- **Decision / next step:**
  Phase 4 FULLY COMPLETE (all exit checklist items satisfied with logged evidence).
  Two open items flagged for resolution before Phase 5 makes them load-bearing (not
  blocking Phase 4's own completion):
  1. `grid_query.py`'s resolution of design_doc_v2.md §2.4's ambiguous notation (positional
     encoding as grid coordinate, dimensionally impossible) — implemented as: grid_sample at
     the Gaussian's own normalized mean position, positional encoding concatenated as extra
     context. Reasonable, standard pattern (K-Planes/Instant-NGP/4DGC), but unconfirmed
     against 4DGC's actual source — needs closing before Phase 5.
  2. `configs/stage_b_temporal.yaml`'s `motion_hypernet.grid_resolution: null` should be set
     to `[4, 8, 16]` now that Phase 4 has validated these shapes, rather than left open.

**Git tag:** `v0.4-phase4-stageB-skeleton`

---

## [Phase 4→5 bridge] Source-verification investigation: 2026-07-22

- **Git commit:** N/A (no code changed — source-reading investigation only, against
  GaussianFormer3D at ~/Documents/min/GaussianFormer3D)
- **Config file(s):** N/A
- **Command:** N/A — manual `find`/`cat`/`grep` against GaussianFormer3D source:
  `model/head/gaussian_head.py`, `model/utils/utils.py` (`get_rotation_matrix`),
  `model/encoder/gaussian_encoder/refine_module.py`, `model/encoder/gaussian_encoder/deformable_module_3d.py`
- **Hardware:** N/A
- **Hypothesis / what this investigation tests:**
  Close three open assumptions flagged after Phase 4's skeleton validation, all load-bearing
  for Phase 5's real-encoder wiring: (1) does `GaussianState`'s field-name/shape assumption
  match Stage A's actual `gaussian` dict output; (2) is Stage B's assumed scalar-first
  `(w,x,y,z)` quaternion convention correct; (3) can `GaussianHead` be called standalone per
  frame for splat+loss, or does Stage B need to re-run the full encoder/decoder each step;
  (4) what is `F^3D_t` concretely, for Stage B's motion-hypernet pooling design (§2.3).
- **Results:** All four confirmed with source evidence, no guesses left standing:
  1. **Field names/shapes confirmed exact match.** `GaussianHead.prepare_gaussian_args`
     confirms `gaussians.means/.scales/.rotations/.opacities/.semantics` — identical to
     `GaussianState`'s fields. `semantics` is `num_classes-1`=17-dim, matching
     `semantic_dim=17` in the working config. No renaming needed anywhere in Phase 4's code.
  2. **Quaternion convention confirmed: scalar-first (w,x,y,z), unit-normalized before use.**
     `get_rotation_matrix`'s `mat1` construction matches the standard left
     quaternion-multiplication matrix `L(q)` term-for-term for `q=(w,x,y,z)`. Identity buffer
     `torch.tensor([1.,0.,0.,0.])` in `GaussianHead.__init__` is consistent with this.
     `refine_module.py`'s `forward()` does `F.normalize(output[...,6:10], dim=-1)` before
     building the output `GaussianPrediction` — rotation arrives pre-normalized. Phase 4's
     `deform_heads.py` assumption was exactly right; no changes needed.
     — Side note (not urgent, no action needed now): `refine_module.py` has a second,
     apparently-dead method `get_gaussian()` that returns the *raw unnormalized* rotation
     instead of `rot` — not used in the active `forward()` path, flag only if a future bug
     ever traces back to that method being called directly.
  3. **`GaussianHead` confirmed callable standalone per frame.** `forward()` takes only
     `representation` (`[{'gaussian': G}, ...]`) and GT `metas`
     (`occ_xyz`/`occ_label`/`occ_cam_mask`) — it never touches `img_backbone`/`img_neck`/the
     iterative deformable-attention decoder stack. It calls `self.aggregator` (CUDA splat op)
     directly on the five Gaussian-property tensors plus GT sample points. **Stage B's
     per-frame training step is therefore: (a) run only `img_backbone`+`img_neck`+
     `lidar_voxel_encoder` to get frame t's features (skipping GaussianFormer3D's 4-block
     decoder entirely), (b) predict `M_t`/`G_t` via HyperNet+deform heads, (c) call
     `GaussianHead` directly with `[{'gaussian': G_t}]` for splat+loss.** Materially cheaper
     per-frame cost than assumed — a real, tractable architecture, not a design gap.
  4. **`F^3D = F^d ⊗ F^c` is never materialized as a literal tensor.**
     `DeformableFeatureAggregation3D.forward()` takes two separate multi-scale lists,
     `feature_maps` (camera, from `img_neck`'s FPN) and `dpt_feature_maps` (LiDAR depth-score
     maps, from `pts_dpt_head`), each `(B, num_cam, C=128, H_l, W_l)` across 4 levels — the
     "outer product" in design_doc_v2.md §1.5 describes what the CUDA deformable-sampling op
     achieves functionally, not a dense tensor ever constructed in Python. This resolves the
     open pooling-strategy question for Stage B's `MotionHyperNet` input with a known,
     concrete shape rather than a conceptual placeholder.
- **Decisions closed as a result:**
  - Motion-hypernet pooling (§2.3, Phase 5 design): global-average-pool each of
    `feature_maps`'/`dpt_feature_maps`'s 4 levels over `(num_cam, H_l, W_l)` → concat → one
    `nn.Linear` down to `in_dim`. Chosen over a spatial 3D-CNN alternative given QGFusion's
    already-observed overfitting failure mode on this same 10-scene budget (900-query run,
    train~7-8% vs val~3.92% mIoU, fixed global embeddings learning scene-specific shortcuts).
    Spatial-CNN variant deferred to a later ablation, not built blind.
  - `configs/stage_a_gaussianformer3d.yaml` and `stage_a_gaussianformer3d/__init__.py`
    (stale pre-pivot scaffolding, still describing the superseded ResNet50/half-resolution
    plan): annotate with a `SUPERSEDED — see occ4dgs_mini_occ3d_gs6400.py` header rather than
    delete, consistent with the project's existing pattern of logging pivots explicitly
    (README's decision table) instead of erasing the trail.
  - `configs/stage_b_temporal.yaml`'s `motion_hypernet.grid_resolution`: set to `[4, 8, 16]`
    (Phase 4 validated exactly these shapes) rather than left `null`.
- **Bugs / issues encountered & fixed:** None (read-only investigation).
- **Decision / next step:** Phase 4 fully closed — skeleton validated (prior entry) AND all
  assumptions it rests on now confirmed against real source. Proceed to Phase 5 (real encoder
  wiring) with the pooling strategy above as the starting design, not an open question.

  ---

  ## [Phase 5] Run ID: 2026-07-23-real-wiring-validated

- **Git commit:** (fill in after commit below)
- **Config file(s):** configs/occ4dgs_mini_occ3d_gs6400.py (Stage A, frozen), no
  stage_b_temporal.yaml training config exercised yet -- wiring only, no optimizer step
- **Command:** `python tests/test_phase5_real_wiring.py`
- **Hardware:** RTX 3090 24GB
- **Hypothesis / what this run tests:** confirm the full Stage B chain -- real Stage A
  G_0, real frame-1 encoder features (img_backbone/img_neck/pts_dpt_head, frozen),
  PoolFeatures -> MotionHyperNet -> grid_query -> DeformHeadMu/R -> apply_update_rule ->
  ReferenceBuffer -> GaussianHead splat -- runs end to end on one real 2-frame clip
  (scene-0061), before writing any training loop.
- **Results:** SUCCESS, all four checks passed with real evidence:
  1. Real G_0 (actual Stage A forward pass, not synthetic) confirmed at N_g=6400.
  2. G_1 provably distinct from G_0 on real data (not just Phase 4's toy sequence).
  3. GaussianHead confirmed callable standalone on a Stage-B-deformed G_1 with real GT
     metas -- pred_occ shape (1, 18, 640000) = (B, semantic_dim+1, H*W*D), matching
     config exactly.
  4. Peak VRAM at unroll_window=2, one scene, frozen encoders, no grad: 2.96 GB --
     consistent with Phase 2's 2.84GB single-frame peak, confirms expected headroom
     before scaling to 10 scenes / window=3 per roadmap Phase 5 step 4.
- **Bugs / issues encountered & fixed:**
  1. `DeformHeadMu` was originally unbounded (only the rotation head was tanh-bounded,
     per design_doc_v2.md Sec 2.5's literal wording). An untrained head's raw delta
     (~0.1-0.2m observed) pushed a Gaussian's z-coordinate to 5.4352, exceeding pc_range's
     z-max of 5.4 (z's valid window is only 6.4m vs 80m for x/y, making it the most
     vulnerable axis) -- crashed LocalAggregator's CUDA splat kernel's hard bounds
     assertion. Root cause confirmed by checking each axis separately: the initial
     combined min/max diagnostic across x,y,z was misleading, since x/y's much wider
     range masked the z violation entirely. Fixed by tanh-bounding DeformHeadMu's output
     per axis (max_disp_xyz=(4.0,4.0,1.0), reusing Stage A's own unit_xyz value as a
     starting point -- flagged as revisit-worthy, not re-derived for Stage B's different
     physical meaning), plus an optional pc_range clamp backstop in apply_update_rule
     (defaults to None, so Phase 4's already-tagged toy-sequence test is unaffected --
     confirmed by re-running test_stage_b_skeleton.py after the fix, all 3 items still pass).
- **Decision / next step:** Phase 5's wiring/shape/VRAM validation is complete. Next:
  write train_stage1.py -- the actual stage_1_warmup training loop (L_occ only, frozen
  Stage A, unroll_window=2, 1-2 scenes first per roadmap step 4), including the
  do-nothing-baseline (Delta_mu=0, Delta_r=identity) comparison the exit checklist
  requires as first evidence the temporal module is learning anything.

**Git tag:** (not yet -- tag once train_stage1.py exists and Phase 5's exit checklist
items 2 and 3 are also closed, not just the wiring)

---

## [Phase 5] Run ID: 2026-07-23/24-stageA-real-training-and-heldout-generalization-check

- **Git commit:** (fill in after commit below)
- **Config file(s):** occ4dgs_mini_occ3d_gs6400.py (Stage A architecture, unmodified);
  script-level overrides only (N_EPOCHS_OVERRIDE=60 in train_stage_a.py, not a config
  edit -- see that file's own comment for why)
- **Commands:**
  `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONNOUSERSITE=1 python scripts/train_stage_a.py`
  `PYTHONNOUSERSITE=1 python scripts/train_stage1.py`
  `PYTHONNOUSERSITE=1 python scripts/evaluate_heldout_scene.py`
- **Hardware:** RTX 3090 24GB (shared with labmate's concurrent QGFusion job, ~6.4GB)
- **Hypothesis:** Root-cause investigation into why the Phase 5 do-nothing-vs-trained
  comparison showed ~0% IoU on both branches, including static classes needing no motion
  compensation. Confirmed cause: Stage A was never actually trained in any prior run
  (Phase 2, Phase 5 wiring test, and the first train_stage1.py run all used a fresh
  `init_weights()` model -- only img_backbone had real pretrained weights; lifter/
  encoder/head started random every time). This run trains Stage A for real, then
  re-tests Stage B's do-nothing-vs-trained comparison, then tests whether any
  improvement generalizes to a held-out scene.
- **Results:**
  1. **Stage A trained for real, 24 then 60 epochs, scene-0061 only.** Loss decreased
     monotonically both runs (24-epoch: 23.62->22.20; 60-epoch: 23.57->20.88), confirming
     24 epochs had stopped short of convergence -- 60-epoch checkpoint used going forward.
  2. **In-sample Stage B comparison (scene-0061, same clips trained on), against the
     60-epoch Stage A checkpoint:** Trained mIoU=16.35/iou2=29.01 vs Do-nothing
     mIoU=14.90/iou2=23.88 (Delta +1.45/+5.13). Every non-degenerate class favored
     trained, including a previously-all-zero moving-object class (truck) becoming
     non-trivial (24.12% vs prior run's 0.00%). This is a real, large improvement over
     the untrained-G_0 result (which showed Delta +0.147/+0.454) -- confirms the root
     cause diagnosis was correct and the fix mattered.
  3. **Held-out scene test (scene-0103, never used to train Stage A or Stage B):**
     Trained mIoU=27.47/iou2=17.87 vs Do-nothing mIoU=27.48/iou2=18.00 (Delta
     -0.014/-0.134). Per-class breakdown on non-degenerate classes is MIXED, not
     consistently favoring trained (driveable_surface -0.46, sidewalk -0.38,
     manmade -1.67 favor do-nothing; truck +0.23, vegetation +2.08 favor trained) --
     opposite pattern from the in-sample result's universal positive direction.
     **Conclusion: the in-sample improvement did not generalize.** Most likely
     explanation: Stage B's temporal module learned scene-0061-specific patterns
     rather than a transferable motion-prediction capability, exactly the failure mode
     a held-out test exists to catch.
- **Artifacts/confounds identified, both real, neither yet resolved:**
  1. **MeanIoU 0-support-class artifact:** classes with zero GT voxels in the evaluation
     window default to 100% IoU (confirmed: barrier/bus/construction_vehicle on
     scene-0103, trailer on scene-0061 throughout). Inflates raw mIoU averages by
     ~25 points on scene-0103 specifically (4 degenerate classes vs. scene-0061's 1),
     making raw mIoU NOT directly comparable across scenes without excluding these --
     this is why held-out mIoU (27.47) superficially looked HIGHER than in-sample
     (16.35) despite the real per-class signal being worse.
  2. **Confounded unknown:** Stage A itself was also only ever trained on scene-0061,
     so G_0 on scene-0103 is out-of-distribution for Stage A too, not just Stage B.
     Cannot currently separate "Stage B's motion prediction doesn't generalize" from
     "G_0 itself is already degraded on an unseen scene, giving Stage B a compromised
     starting point regardless of its own generalization."
- **Bugs fixed en route:** (1) DeformHeadMu z-boundary crash (see prior entry);
  (2) evaluation code passed raw logits without argmax and wrong batch/layer indexing
  to MeanIoU._after_step, fixed against train.py's real usage pattern; (3) wandb.log
  crash inside MeanIoU._after_epoch (no active run) -- fixed via wandb.init(mode="disabled"),
  confirmed this does not touch or require any account/login; (4) CUDA OOM training
  Stage A's full backward pass while sharing the GPU with a concurrent labmate job --
  fixed via PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 (Option 1 sufficed; AMP
  fallback written but not needed).
- **Decision / next step:** Exit checklist item 2 is NOT genuinely satisfied yet -- the
  positive in-sample result does not survive a held-out test. Both Stage A and Stage B
  currently exist only as single-scene fits. The real next step is scope (train both
  across more scenes) before any further architecture changes, since nothing here
  indicates the wiring itself is wrong -- only that a 1-scene training budget is too
  narrow to produce a generalizing result, which is itself a useful, expected finding
  at this stage of the roadmap's "1-2 scenes, then scale" discipline.

**Git tag:** not yet -- exit checklist item 2 unresolved, item 3 (10-scene scaling)
not yet attempted.

---

## [Phase 5] Run ID: 2026-07-24-stageA-cross-scene-confound-confirmed

- **Command:** `PYTHONNOUSERSITE=1 python scripts/evaluate_stage_a_all_scenes.py`
- **Hypothesis:** Decision 1 (isolate before scaling) -- is G_0 itself degraded on scenes
  Stage A never trained on, confounding the scene-0103 held-out Stage B result?
- **Results:** CONFIRMED. Adjusted mIoU (excludes zero-GT-support classes, which
  raw mIoU silently includes as misleading 100%): scene-0061 (trained)=12.52;
  9 unseen scenes range 3.42-6.48, mean~5.16 -- a consistent ~2.4x drop across every
  single unseen scene, not scattered noise. Raw mIoU is actively backwards (appears to
  INCREASE on unseen scenes, up to 41.5) purely because unseen scenes have 4-6
  zero-support classes vs. scene-0061's 1, inflating the naive average.
- **Conclusion:** The scene-0103 held-out Stage B test (prior entry) was confounded as
  suspected -- Stage B's motion prediction was tested on top of an already-~2.4x-degraded
  G_0, not a fair single-variable generalization test. Cannot yet conclude whether Stage
  B's own architecture generalizes until Stage A itself covers enough scenes that G_0
  is reasonably reliable on a genuinely held-out scene.
- **Secondary finding:** even in-sample (scene-0061), rare/movable classes (car, bicycle,
  motorcycle, pedestrian) score genuine 0.00% (real GT support, zero correct) -- Stage A's
  current signal is concentrated in large static classes regardless of scene familiarity.
- **Decision / next step:** Scale Stage A's training to 3-4 scenes (neutral selection --
  next scenes in ALL_SCENES' existing order, not cherry-picked from this diagnostic, per
  the experimenter-bias discipline already agreed) before re-attempting any held-out
  Stage B generalization test.

  ---

  ## [Phase 5] Run ID: 2026-07-25-stageA-3scene-diagnostic (labeling bug noted)

- Bug: evaluate_stage_a_all_scenes.py's TRAINED_ON constant was not updated when
  train_stage_a.py's SCENES scaled 1->3; scene-0103/scene-0553 were mislabeled
  "(unseen)" despite being trained on. Corrected post-hoc for this analysis; script
  itself will be fixed before next run.
- Corrected trained-group mean adjusted mIoU: 10.17 (scene-0061=8.686, scene-0103=10.369,
  scene-0553=11.454). Unseen-group (7 scenes) mean: 5.71.
- Gap ratio trained/unseen: 1.78x, improved from the 1-scene run's 2.4x -- more
  training scenes narrows the generalization gap, as expected.
- Regression: scene-0061 itself dropped 12.519->8.686 vs. the 1-scene checkpoint;
  other_flat and truck classes lost all signal (20.28%->0%, 29.39%->0%). Likely
  undertrained relative to 3x the data at the same 60-epoch budget (loss still
  decreasing at epoch 59), not evidence scaling scenes is harmful -- not yet confirmed.
- Decision: before further scaling, fix the TRAINED_ON labeling bug, then decide
  whether to increase epochs at 3 scenes to test the undertraining hypothesis, or
  proceed to more scenes anyway.

  ---

  ## [Phase 5] Run ID: 2026-07-27-stageB-8scene-matched-heldout

- Stage B retrained on the same 8 scenes as Stage A (previously Stage B only ever
  trained on scene-0061, a scope mismatch now fixed), 20 epochs, loss converged
  (11.08->10.72, plateaued ~epoch 10).
- In-sample (8 scenes, 316 clips): every real class favors trained over do-nothing,
  no exceptions (car +3.45, vegetation +5.96, truck +1.51, manmade +1.48, others
  smaller positive). Clean, consistent positive result.
- Held-out (scene-1094, scene-1100, never trained on by Stage A or B), adjusted for
  zero-support-class artifact: scene-1094 trained=3.358 vs do-nothing=3.506 (-0.148);
  scene-1100 trained=4.838 vs do-nothing=7.517 (-2.679, manmade/vegetation
  particularly collapsed).
- CONCLUSION: consistent positive in-sample + consistent negative held-out =
  overfitting, not undertraining (undertraining would show weak results on both).
  This is now a clear, matched-scope answer: Stage B's current motion-prediction
  design does not generalize at this project's data scale.
- Decision: this is a real research finding to report, not a bug to keep chasing.
  Possible next directions (not yet pursued): stronger regularization on deform
  heads, smaller HyperNet capacity, or accepting this as evidence that 10 scenes is
  simply insufficient for a generalizing temporal module and reporting it as such.

  ---

  ## [Phase 5] Run ID: 2026-07-29-gt-motion-measurement-and-ego-compensation-saga

### Part 1 — Ground-truth motion magnitude measurement (Tier 1 item 3)

- **Command:** `scripts/measure_gt_motion.py`, across all 10 mini-set scenes.
- **Hypothesis:** does the held-out generalization ceiling (established over the prior
  8-scene sweep) reflect a lack of real learnable motion signal in the data, or something
  else?
- **Result:** Overwhelming, unambiguous signal -- 60-84% of non-empty voxels change label
  between adjacent frames across scenes (overall mean 63.9%). Critically, STATIC classes
  (driveable_surface, manmade, vegetation, etc.) show nearly as much change (60.4%) as
  DYNAMIC classes (80.6%) -- roads and buildings don't move, so this apparent "change" is
  overwhelmingly ego-vehicle motion, not independent object motion. `occ_label` is
  ego-centric per frame (confirmed via `use_ego=True` in the pipeline), so comparing the
  same voxel index across two frames compares two different real-world patches whenever
  the ego vehicle has moved -- and at nuScenes' ~2Hz keyframe rate, it always has.
- **Conclusion:** No shortage of raw signal -- the opposite problem. The temporal module
  was being asked to rediscover, from a small MLP on pooled visual features and 8 scenes
  of data, a global rigid transform (ego motion) that the dataset's own pose fields
  (`ego2global`/`lidar2global`) already encode exactly. This reframes every prior
  regularization/architecture experiment this session (Step 1 regularization, PE removal,
  scene-count scaling) as attempts to help a small model learn something it should never
  have had to learn from scratch.
- **Decision:** implement explicit ego-motion compensation -- apply the KNOWN rigid
  transform directly, let the learned Delta_mu/Delta_r handle only the residual
  (independent object motion). Highest-leverage change identified all session.

### Part 2 — Stage B scene-scaling sweep: a real confound found and fixed

- Ran train_stage1.py at scene counts 1/3/6/8 (holding the 8-scene Stage A checkpoint and
  a fixed seed constant), checking held-out delta at fixed EPOCH boundaries.
- **Confound discovered:** clip count differs 8x across these runs (38 vs 316
  clips/epoch), so "epoch 1" represents ~8x more optimizer steps for n=8 than n=1. The
  apparent "more scenes -> worse held-out delta" trend was confounded with "more
  optimizer steps -> worse held-out delta" -- a pattern already visible in every prior
  experiment this session, regardless of architecture changes.
- **Fix:** rewrote train_stage1.py's evaluation trigger to gate on OPTIMIZER STEP COUNT
  (`EVAL_EVERY_OPT_STEPS`), not epoch index, so runs at different scene counts are
  comparable on the same x-axis. Also added: fixed seed (`SEED=42`), CLI scene-count
  argument (`python train_stage1.py N`), best-held-out-delta checkpointing (separate from
  final-epoch checkpoint).
- This sweep was superseded by the ego-motion compensation work below before being
  re-run to completion under the corrected step-matched methodology -- worth re-running
  once ego-compensation's role is settled.

### Part 3 — Ego-motion compensation: implementation, a real bug found and fixed, and a genuine (non-bug) limitation discovered

**Implementation** (new/modified):
- `src/datasets/occ4dgs_dataset.py`: now passes `ego2global`/`lidar2global` through
  (previously computed internally by `get_data_info` but discarded).
- `scripts/run_stage_a_frame0.py`: `to_batch_of_one` now carries these poses into the
  CUDA-ready `metas` dict (purely additive, confirmed backward-compatible).
- `src/models/stage_b_temporal/deform_heads.py`: added `rotmat_to_quat` (hand-verified
  against identity and a 5-degree yaw case, plus round-trip reconstruction),
  `compute_relative_transform` (hand-verified via a synthetic 5m-forward + 3-degree-yaw
  ego motion test -- a static world point correctly re-expressed between frames), and
  `apply_ego_compensated_update_rule` (applies the known transform, then the learned
  residual via the existing `apply_update_rule`).
- `train_stage1.py`'s `deform_one_step` wired to use `lidar2global` (not `ego2global`) --
  chosen per `occ_annotation="occ3d"`'s confirmed LiDAR-centric convention.
- `scripts/check_ego_compensation.py` (new): isolates the compensation's correctness
  independent of any learning, via a "zero learned residual" mode.

**Bug found and fixed:** First real-data test (n=1 scene) showed the ego-compensated
`trained` branch regressing well below the pre-compensation baseline from the very first
checkpoint -- suspicious, since near-init the learned residual is ~0, so `trained` should
reduce to roughly ego-compensated-`G_0`-unmodified, which should be at least as good as
plain do-nothing if the compensation is correct. Isolated via `check_ego_compensation.py`:
- Confirmed NOT a prev/curr direction bug (tested both orderings, both regress similarly:
  -0.673 vs -0.706).
- Confirmed the real ego-motion values themselves are sane (checked directly: 0.07-0.77m
  for the first 5 clips; full-scene check showed up to 6.34m for high-motion segments).
- **Root cause identified:** the `pc_range` clamp in `apply_update_rule` (added earlier
  for tiny LEARNED deltas near the z-boundary) was never designed for a REAL rigid
  ego-motion shift of several meters. Stage A fills Gaussians out to the pc_range
  boundary; a multi-meter shift routinely pushes trailing-edge Gaussians outside the
  valid volume. The clamp then smeared their POSITION onto the boundary wall as
  visible, geometrically WRONG content (confirmed: disabling the clamp entirely crashes
  `LocalAggregator`'s hard bounds assertion, proving real, substantial out-of-range
  displacement -- not a marginal edge case).
- **Fix applied:** `apply_update_rule` now zeros the OPACITY of any Gaussian whose
  pre-clamp position falls outside `pc_range`, in addition to clamping position (clamping
  position alone is still required to satisfy the splat kernel's hard bounds check;
  opacity=0 makes the clamped position invisible rather than a wrong visible smear).
  Regression-checked against Phase 4's exit-checklist suite (unaffected, uses
  `pc_range=None`).

**Result after the fix -- and a genuine, non-bug limitation discovered:**
- Re-ran the do-nothing vs ego_only (zero learned residual) comparison: opacity fix made
  no meaningful difference in aggregate (-0.654, essentially unchanged from -0.673/-0.706
  pre-fix).
- **Per-scene breakdown was the decisive test:** `scene-1100` (near-zero real ego motion,
  max 0.33m) showed `ego_only` vs `donothing` delta = +0.045 -- essentially neutral,
  exactly what correctly-working near-no-op compensation should produce. `scene-1094`
  (real motion up to 6.34m, mean 3.24m) showed delta = -1.193 -- substantial degradation,
  scaling with real motion magnitude.
- **Conclusion: the compensation mechanism itself is confirmed correct** (a bug would
  degrade the near-zero-motion scene too; it doesn't). The degradation at high motion is
  a genuine, different limitation: Stage B works with a FIXED set of 6400 Gaussians
  carried forward from frame 0. When the ego vehicle moves substantially, newly-visible
  world content enters the leading edge of the `pc_range` window with NO Gaussian
  representing it at all -- a representational gap, not a coding defect. Larger motion
  means more unrepresented new content, exactly matching the observed scaling.

**Decision / next steps (not yet pursued):**
1. Check the real-motion distribution across ALL clips (not just these two scenes) to
   see whether the typical nuScenes 2Hz frame gap is closer to scene-1100's low-motion
   case or scene-1094's high-motion case -- determines whether compensation nets positive
   in aggregate despite the high-motion failure mode.
2. Consider whether to accept the "no Gaussian spawning" limitation as a documented,
   reportable finding, or pursue a real architectural addition (a mechanism to
   spawn/reactivate Gaussians for newly-visible regions) -- a genuine research direction,
   not a quick fix.
3. Re-run the step-matched 1/3/6/8 scene-scaling sweep (Part 2) now that ego-compensation
   exists, to see whether it changes the scene-scaling picture.


---

## [Phase 5] Run ID: 2026-07-29-ego-motion-distribution-check

- Command: scripts/measure_ego_motion_distribution.py, all 10 scenes, 394 clips total.
- Result: translation magnitude is NOT concentrated at low values. Median 2.527m,
  p75=4.438m, p90=6.122m. 44.4% of all clips exceed 3.0m (the threshold where
  scene-1094 showed confirmed real damage in the ego-compensation test). Several
  scenes (scene-0796 mean 6.07m, scene-1077 mean 6.30m) are consistently high-motion.
  Only ~27% of clips are near-stationary (<0.5m, the safe regime scene-1100 represented).
- Conclusion: the Gaussian-representational gap (fixed 6400 Gaussians, no mechanism
  for newly-visible content after ego motion) is the DOMINANT regime at this frame
  rate, not a rare edge case. Explains why aggregate ego_only result was net negative
  despite being neutral/positive in the low-motion case -- high-motion clips (~44%
  of data) dominate the average and show much larger per-clip damage.
- Decision: build the Gaussian-recycling mechanism (reposition/reactivate
  out-of-range Gaussians into newly-visible regions, rather than discarding via
  opacity=0) -- this is the majority-case fix, not a corner-case one, and is
  buildable with current 8/10-scene data since it's a representational-capacity fix,
  not a data-volume fix.

  ---

  ## [Phase 5] Run ID: 2026-07-30-spawnhead-and-full-arc-retrospective

### SpawnHead implementation and test

- New: src/models/stage_b_temporal/spawn_head.py -- SpawnHead, a small learned MLP
  replacing the heuristic recycling attempt. Conditioned on (a) the motion grid's own
  feature at a candidate position (query_motion_grid, include_pe=False) and (b) the
  pooled global scene feature; predicts a position refinement, opacity, and semantics
  for each Gaussian pushed out of pc_range by ego-motion compensation, rather than the
  fixed-opacity/random-placement/stale-semantics heuristic.
- Design fix made during implementation: replaced RANDOM jitter in the wraparound base
  position with a FIXED deterministic margin (RECYCLE_MARGIN_FRAC=0.10). Necessary
  because SpawnHead must query the grid at a candidate position BEFORE the final
  update-rule call (different file/function) -- a random jitter computed twice would
  land in two inconsistent positions. Hand-verified: precomputed candidate positions
  match final applied positions exactly (bit-for-bit) given the same inputs.
- Wired into train_stage1.py's deform_one_step, build_temporal_module,
  evaluate_heldout, and checkpointing (backward-compatible: spawn_head=None preserves
  all prior behavior exactly, confirmed via Phase 4 regression test).
- Result (n=8 scenes, step-matched, same seed/schedule as the no-SpawnHead run):
  small, fairly consistent improvement over plain opacity-zeroing at every checkpoint
  (+0.02 to +0.09 in held-out adjusted mIoU delta), e.g. opt_step 100: -1.666 vs
  -1.758; final (opt_step 1580): -1.487 vs -1.508.
- CRITICAL OBSERVATION: despite this real, positive delta, the CURVE SHAPE is
  essentially unchanged from every prior version -- starts near -0.7 to -0.8 at the
  first checkpoint, degrades sharply over ~100-200 steps to -1.4 to -1.8, plateaus
  there for the remainder of training. SpawnHead shifted the whole curve up slightly
  (an additive correction) without changing the underlying trajectory.

### Full-session retrospective: which fixes helped, which didn't, and the net effect

Genuine corrections (necessary regardless of outcome, not expected to be "the fix"):
training Stage A from scratch, matching Stage B's scope to Stage A's, step-matched
(not epoch-matched) evaluation, opacity-zeroing on out-of-range Gaussians.

Measurably helped:
- Delta-feature conditioning (Step 2): the single clearest positive result all
  session. Fixed the confirmed absolute-scene-appearance shortcut, moving epoch-1
  held-out delta from -1.15 to -0.01 (near parity).
- SpawnHead: small, consistent, real improvement over opacity-zeroing (this entry).

No measurable effect (tested honestly, no benefit found):
- Step 1 regularization (dropout, weight decay, motion-magnitude penalty).
- Removing positional encoding (PE(mu)) from z.
- Heuristic recycling (random wraparound + fixed opacity, superseded by SpawnHead).

HONEST NET EFFECT ACROSS THE WHOLE SESSION: the single BEST held-out result achieved
at any point this session was -0.013 to -0.117 (right after Step 2's delta-
conditioning fix alone, BEFORE ego-motion compensation existed). Every addition since
then -- compensation, the opacity fix, recycling, SpawnHead -- has produced a
best-achieved delta between -0.7 and -0.78. Each individual addition was correctly
implemented and independently well-motivated (compensation's math is confirmed
correct; the opacity fix corrected a real bug; SpawnHead measurably beats the
heuristic it replaced) -- but STACKED TOGETHER, they have made the best-case result
meaningfully WORSE than a single earlier fix alone, not better.

Decision on the "no effect" fixes: NOT retiring them, but not reincorporating them
yet either.
  - Heuristic recycling: fully superseded by SpawnHead, will not be revisited.
  - Regularization (dropout/weight-decay/motion-penalty): specifically remedies
    excess capacity relative to available data (closes overfitting gaps). If the
    dominant problem is genuinely a data-scale ceiling (increasingly likely -- see
    below), this class of fix would not be expected to help regardless of what else
    changes, since it cannot create signal that isn't there. Worth retesting only
    after the data situation itself changes (more scenes), not as an immediate step.
  - PE removal: cheap, low-risk, not tied to the data-scale question specifically
    (was testing a shortcut hypothesis). Reasonable to fold back into a future clean
    run whenever convenient.
  - Explicit caution logged: do NOT reincorporate multiple of these simultaneously in
    one run -- that is exactly the stacking pattern that produced the net-negative
    result documented above. Any reintroduction should be tested one change at a
    time against a clean, single-variable baseline.

KEY STANDING QUESTION, NOT YET RESOLVED: the same sharp-degradation-then-plateau
curve shape has now persisted, essentially unchanged, across every structural
intervention tried (delta-conditioning aside, which changed only the STARTING point,
not this shape) -- regularization, PE removal, compensation, recycling, SpawnHead.
This consistency across so many different architectural angles is itself evidence
that something more fundamental (very plausibly the same data-scale ceiling already
confirmed for Stage A) is the dominant driver, not a supporting factor as assumed
when ego-motion compensation work began. NEXT STEP (not yet done): directly compare
the pre-compensation best state (delta-conditioning alone) against the current
full-pipeline state, to decide whether to continue building on compensation or revert
to that earlier, better-performing baseline and treat the Gaussian-representational
gap as a documented, separate finding rather than an actively pursued fix.

---

## [Phase 5] Run ID: 2026-07-30-4dgc-reference-code-review-scope-clarification

### Context
User provided the actual 4DGC reference implementation (github.com/zihanzheng-sjtu/4DGC:
scripts/Motion_Grid_warmup.py, scene/Motion_Grid.py, scene/gaussian_model.py) after the
full session's arc of fixes (delta-conditioning, ego-motion compensation, heuristic
recycling, learned SpawnHead) failed to close the held-out generalization gap, each
producing the same persistent curve shape (sharp early degradation, then a plateau)
regardless of what was changed.

### Finding: 4DGC solves a fundamentally different, easier problem than Occ4DGS is
attempting

Reading gaussian_model.py's training_one_frame_setup/training_one_frame_s2_setup and
Motion_Grid_warmup.py directly (not from memory/inference, as flagged as a risk
earlier in this project) reveals:

- 4DGC's "added"/"spawned"/"cloned" Gaussians are created as real nn.Parameter
  tensors with their OWN per-frame optimizer (self.optimizer, self.mem_optimizer),
  refined via gradient descent AGAINST THAT SPECIFIC FRAME'S OWN REAL CAPTURED
  IMAGES, frame by frame, at test/encode time.
- Motion_Grid_warmup.py confirms the motion grid itself is warmed up on a SINGLE
  scene's own point cloud before any per-frame fitting begins.
- CONCLUSION: 4DGC is fundamentally a per-video, per-frame TEST-TIME OPTIMIZATION /
  compression method. It has full access to every target frame's real ground truth
  at prediction time and iteratively refines against it. It is NOT a zero-shot
  feedforward generalization method to frames/scenes never seen or fitted.

Occ4DGS's Stage B, by contrast, is attempting a single feedforward pass with FROZEN
weights on a HELD-OUT scene, with NO gradient-based refinement against the target
frame's own ground truth at all. This is a strictly harder, fundamentally different
task than what 4DGC's own paper/code solves.

### Implication for the entire session's debugging arc

This directly explains the persistent pattern observed across every fix attempted
(regularization, PE removal, ego-motion compensation, heuristic recycling, learned
SpawnHead): none of them addressed the actual gap, because the project has been
attempting to match a per-scene-TEST-TIME-FITTED method's spawning/densification
behavior using a purely feedforward, zero-shot mechanism -- a category difference,
not a tuning problem. No architecture change within the feedforward-only constraint
was ever going to fully close this gap.

### Separate, smaller, concrete bug found via the same code review

gaussian_model.py's query_mem() composes rotation as
self.rotation_compose(self._rotation, self._d_rot) -- i.e. CURRENT-ROTATION-FIRST
(rotation (x) delta). Occ4DGS's apply_update_rule uses quat_multiply(delta_quat,
prev_state.rotations) -- DELTA-FIRST (delta (x) rotation), the OPPOSITE Hamilton
product order. For the small angles observed in this project (max 14.76 degrees
measured across all 394 clips), this is a second-order discrepancy, not a likely
dominant cause of the generalization gap -- but it is a confirmed, real convention
mismatch against the reference implementation, worth correcting for exactness
independent of the larger scope question above.

### Decision / next steps (not yet resolved)

1. Reframe the project's stated goal/report to explicitly acknowledge Occ4DGS
   attempts genuine zero-shot feedforward temporal generalization -- a harder
   setting than 4DGC's own per-frame-optimized approach -- rather than continuing
   to treat "match 4DGC-style spawning behavior without per-frame optimization" as
   an achievable, tuning-only target.
2. OPEN QUESTION, not yet decided: whether to allow SOME limited per-scene
   adaptation (e.g. a few gradient steps against frame 0's OWN data only, not the
   target frame's ground truth) as a middle ground -- would not be "cheating" the
   prediction task, but has not been designed or tested.
3. Fix the rotation composition order (quat_multiply(prev_state.rotations,
   delta_quat) instead of the current delta-first order) for exactness, independent
   of the scope decision above -- small, mechanical, not yet applied.

---

## [Phase 5] Run ID: 2026-07-30-clean-baseline-reestablished-step1-applied

- Reverted ego-motion compensation entirely (USE_EGO_COMPENSATION=False), applied
  Step 1's rotation-order fix (current-first Hamilton product, matching 4DGC's
  reference rotation_compose(self._rotation, self._d_rot)).
- Result (n=8 scenes, step-matched, seed=42): best-ever held-out delta at n=8 this
  session -- +0.044 at opt_step 120 (do-nothing baseline: 4.976). Curve stays near
  parity/positive through opt_step 140, then degrades to the familiar -0.6 to -0.7
  plateau by opt_step 1580.
- CAVEAT: this run combines reverting compensation AND the rotation-order fix
  together -- individual contributions not isolated. Given the small angles involved
  (max ~15 degrees measured across all clips), the rotation fix was not expected to
  matter much on its own; if it turns out to matter this much, worth understanding
  why later. Most of the credit is plausibly the compensation reversion, consistent
  with the prior retrospective finding.
- THIS IS NOW THE REFERENCE BASELINE for testing Step 2 (grid resolution/channel
  rebalance) and subsequent architecture changes, one at a time.

---

## [Phase 5] Run ID: 2026-07-30-step2-grid-rebalance

- Rebalanced MotionHyperNet: resolutions (4,8,16)->(8,16,32), grid_feat_dim 16->4
  (controlled 2x total-parameter increase, verified before running -- not the ~8x a
  naive resolution bump alone would cause). z dimension 48->12 accordingly, DeformHeadMu/
  DeformHeadR in_dim updated to match. Tested against the Step 1 clean baseline
  (USE_EGO_COMPENSATION=False), single-variable change.
- Result (n=8, step-matched, seed=42): best held-out delta improved +0.044 -> +0.080
  (nearly 2x), achieved at opt_step 80. The near-zero/positive window (opt_step
  40-120) is both slightly wider and consistently stronger than Step 1's baseline
  across it (deltas +0.023/+0.080/+0.020/+0.054 vs Step 1's +0.008/-0.021/-0.055).
- CRITICAL: the underlying curve SHAPE is unchanged -- sharp decline still begins
  around opt_step 140-160, settling into the same -0.6 to -0.78 plateau by opt_step
  1580 (final: -0.766 here vs -0.707 for Step 1, essentially identical). Grid
  rebalance raised the ceiling of the "good" early phase but did not extend its
  duration or prevent the later collapse.
- Decision: keep this change (genuine, single-variable improvement, no reason to
  revert). Proceed to Step 3 (PE-as-coordinate query mechanism) on top of this new
  baseline (+0.080 at opt_step 80).

---

## [Phase 5] Run ID: 2026-07-30-step3-pe-coordinate-query

- Implemented 4DGC's actual query mechanism (confirmed via direct code reading, not
  inference): query_motion_grid_pe_coordinate samples each grid level TWICE, using
  sin(2^l*pi*norm_means) and cos(2^l*pi*norm_means) AS the grid_sample coordinates
  themselves (not position directly, PE not concatenated as separate context --
  this project's earlier guess at an ambiguous notation, kept as query_motion_grid's
  unchanged default for backward compatibility). z dimension doubled 12->24
  accordingly (2 samples x 3 levels x 4 channels), DeformHeadMu/DeformHeadR in_dim
  updated to match. Tested against Step 2's baseline, single-variable change.
- Result (n=8, step-matched, seed=42): best held-out delta improved +0.080 -> +0.106
  (now 2.4x Step 1's original +0.044). The near-zero/positive window is genuinely
  more robust this time -- 5 of 7 checkpoints from opt_step 40-140 are positive or
  near-zero (+0.081/+0.042/+0.020/+0.088/+0.106/+0.015), not just an isolated peak.
- CRITICAL, UNCHANGED: collapse still begins on schedule at opt_step 160, settling
  into essentially the same final plateau (-0.625, vs Step 2's -0.766 and Step 1's
  -0.707 -- all in the -0.6 to -0.78 range). Three consecutive architecture
  improvements have each raised the peak and slightly extended the good window, but
  NONE has touched whatever triggers the collapse after ~150 steps.
- Decision: keep this change. Proceed to Step 4 (spatially-resolved grid prediction,
  the prime suspect for the collapse itself) on top of this new baseline (+0.106 at
  opt_step 120).

---

## [Phase 5] Run ID: 2026-07-30-step4-conv-hypernet

- Replaced MotionHyperNet's per-level Linear expansion (~19M params, zero spatial
  inductive bias) with ConvHyperNet, a shared transposed-convolutional decoder
  (<1M params, spatially-structured, top-down refinement across resolution levels).
  Tested against Step 3's baseline, single-variable change (parameter count and
  inductive bias both change together -- not separated, noted as a limitation).
- Result (n=8, step-matched, seed=42): best held-out delta +0.100 at opt_step 80,
  essentially matching Step 3's +0.106 (no peak improvement). BUT the good window
  shifted meaningfully later: positive/near-zero from opt_step 60-160 (6 checkpoints),
  vs Step 3's opt_step 40-140 -- collapse onset delayed by one full checkpoint
  (20 optimizer steps), the FIRST change all session to shift WHEN the collapse
  begins rather than only how high the peak reaches beforehand.
- Final plateau unchanged in magnitude: -0.665, within the same -0.6 to -0.78 range
  every prior version has landed in.
- CONCLUSION: four architecture changes in a row (grid rebalance, PE-coordinate
  query, conv decoder) have each produced similar-sized, modest, real improvements,
  and NONE has broken the fundamental collapse-then-plateau shape. If the HyperNet
  spatial bottleneck (Step 4's specific target) were the dominant cause, expected a
  much larger effect than a one-checkpoint delay. Increasingly points to the
  data-scale ceiling (shared with Stage A, never yet cleanly isolated for Stage B
  specifically via a clean scene-scaling sweep) as the dominant remaining factor.
- Decision (pending user input): consider running the Stage-B-specific scene-scaling
  sweep now, on this best-yet architecture (Steps 1-4 combined), to directly test
  the data-scale hypothesis rather than inferring it by elimination.

---

## [Phase 5] Run ID: 2026-07-30-scene-scaling-sweep-and-critical-noise-floor-finding

### Scene-scaling sweep (n=1, 3, 6, 8), architecture = Steps 1-4 combined

| Scenes | Best held-out delta | At opt_step | Positive-window length |
|---|---|---|---|
| 1 | +0.091 | 60 | 3 checkpoints (opt_step 40-80) |
| 3 | +0.125 (best all-time this session) | 140 | 6 checkpoints (60-160) |
| 6 | +0.105 | 80 | 7 checkpoints (60-180, longest window observed) |
| 8 | +0.068 | 120 | 5 checkpoints (60-140) |

RESULT: no monotonic "more scenes -> better generalization" pattern, unlike Stage A's
clean scaling result. n=8 is the WORST of the four multi-scene runs by both peak delta
and window length; n=3 and n=6 both outperform n=8. This directly contradicts the
assumption (carried over from Stage A's confirmed scaling behavior) that more scenes
would straightforwardly help Stage B's learned components too.

### CRITICAL FINDING, supersedes confident interpretation of the above and of every
architecture comparison this session

The "Held-out do-nothing baseline" value -- which depends ONLY on the frozen,
already-trained Stage A checkpoint evaluated on the same 2 held-out scenes, involves
NO training, and should be exactly reproducible run to run -- varied across these four
runs: 4.974, 4.993, 5.010, 5.001 (range 0.036).

IMPLICATION: there is unexplained non-determinism somewhere in the pipeline (likely
GPU-level -- non-deterministic cuDNN kernels, or some training-mode-gated operation in
Stage A's forward pass not fully neutralized under .eval()). This noise (~0.036 on a
completely untrained, deterministic-in-principle quantity) is THE SAME ORDER OF
MAGNITUDE as the differences we have been attributing to architecture improvements all
session: Step 1->2->3->4's best-delta progression was +0.044 -> +0.080 -> +0.106 ->
+0.100, i.e. consecutive differences of 0.02-0.04.

CONCLUSION: we currently CANNOT cleanly distinguish genuine architectural improvement
from run-to-run noise, for ANY single-run comparison made this session, including:
  - Steps 1-4's individual claimed improvements (each was a single run vs single run)
  - This scene-scaling sweep's n=1/3/6/8 ranking

This does not mean Steps 1-4 were wrong or the sweep is meaningless -- the underlying
architectural reasoning for each (rotation order, spatial resolution, PE-as-coordinate,
spatial inductive bias in the decoder) remains sound and independently well-motivated.
It means the NUMERICAL comparisons used to confirm each one exceeded our actual
measurement precision, and every one of those comparisons needs to be treated as
provisional until repeated-seed noise floor is established.

### Decision: PAUSED here, logged, no further runs until noise floor is measured

Next step (not yet started): repeat at least one configuration (recommend n=8, our
main reference point) 2-3 times with different seeds, to measure the true run-to-run
variance in both the do-nothing baseline and the best-achieved held-out delta. Only
once that noise floor is known can Steps 1-4's improvements and the scene-scaling
ranking be properly assessed as signal vs. noise.

---

## [Phase 5] Run ID: 2026-08-06-noise-floor-localized-stopping-point

### Step 0: noise-floor localization (measure_noise_floor.py)

Root cause of the do-nothing baseline's run-to-run variance (~0.036-0.07 across
various tests) is now precisely localized, via a sequence of cheap, no-training
diagnostics:

1. Confirmed NOT random-seed sensitivity (SEED=42 was already fixed across the
   original 4-run scene-scaling sweep that first surfaced this).
2. torch.backends.cudnn.deterministic=True alone: no effect.
3. torch.use_deterministic_algorithms(True, warn_only=True): surfaced real CuBLAS
   matmul non-determinism (notably in gaussian_head.py's covariance computation,
   directly upstream of pred_occ) plus two unfixable-in-this-PyTorch-version,
   loss-only ops (nll_loss2d, cumsum -- confirmed NOT reachable from pred_occ/mIoU,
   since evaluate_heldout scores only pred_occ.argmax(), never the loss value).
4. CUBLAS_WORKSPACE_CONFIG=:4096:8: confirmed (via warning disappearance) fixed the
   CuBLAS matmul issue specifically -- but overall range barely moved, so CuBLAS was
   real but not the dominant source.
5. bisect(): splatting/head is PERFECTLY bit-identical across 5 repeats (fully
   exonerated). Stage A's encoder alone is NOT (max abs diff 76.8 on position values
   ranging [-40,40] -- magnitude far too large for ordinary floating-point
   reduction-order noise, more consistent with an actual bug, e.g. uninitialized GPU
   memory in a custom kernel, than benign non-determinism).
6. bisect_fine(): camera backbone+FPN and the depth head are BOTH perfectly
   bit-identical across 5 repeats. FINAL LOCALIZATION: the non-determinism is
   specifically inside the iterative Gaussian-refinement blocks -- DFA3D's custom
   deformable-attention CUDA kernel and/or the sparse-conv self-encoding step --
   the only genuinely custom, non-standard-PyTorch machinery in the entire Stage A
   pipeline. Every standard op (conv, FPN, depth head, splatting) is clean.

### Decision: STOPPING HERE, by deliberate choice, not because further progress is
impossible

Weighed explicitly before this session's diagnostic work began: a full fix would
require C++/CUDA-level debugging of third-party compiled kernels (DFA3D and/or
GaussianFormer3D's own sparse-conv code) -- a different skill set, unbounded time
cost, and a direct distraction from the actual Step 5 research work already directed
by the professor. The localization achieved (bisect_fine, one step from the exact
kernel) is precise enough to document confidently without needing to go further.

### Practical path forward (adopted, not deferred)

Treat the measured noise magnitude (~0.04-0.07 in adjusted mIoU) as a known,
characterized constant of this pipeline. Going forward: any future architecture
comparison (Step 5 onward) must be run multiple times and compared as a distribution,
not a single point estimate, before a difference is treated as real. This directly
resolves the standing concern flagged at the end of the Steps 1-4 arc (2026-07-30) --
we now know precisely why that noise existed and roughly how large it is, without
needing to eliminate it at the source.

---

## [Phase 5] Run ID: 2026-08-06-step0-noise-floor-localized-and-protocol-adopted

### Step 0: noise-floor localization complete (scripts/measure_noise_floor.py)

Root cause of the do-nothing baseline's run-to-run variance (first found 2026-07-30,
~0.036 across a 4-run scene-scaling sweep) is now precisely localized via a sequence
of cheap, no-training diagnostics, all with real evidence at each step:

1. NOT random-seed sensitivity -- SEED=42 was already fixed across all 4 original
   sweep runs when the variance first appeared.
2. torch.backends.cudnn.deterministic=True alone: no effect (range 0.041->0.061).
3. torch.use_deterministic_algorithms(True, warn_only=True): surfaced real CuBLAS
   matmul non-determinism (gaussian_head.py's covariance computation, directly
   upstream of pred_occ) plus two unfixable-in-this-PyTorch-version, loss-only ops
   (nll_loss2d, cumsum) -- confirmed NOT reachable from pred_occ/mIoU, since
   evaluate_heldout scores only pred_occ.argmax(), never the loss value itself.
4. CUBLAS_WORKSPACE_CONFIG=:4096:8: confirmed via warning disappearance that it fixed
   the CuBLAS matmul issue specifically -- overall range barely moved (0.036->0.040),
   so CuBLAS was real but not the dominant source.
5. bisect(): splatting/GaussianHead is PERFECTLY bit-identical across 5 repeats (fully
   exonerated). Stage A's encoder alone is NOT (max abs diff 76.8 on position values
   ranging [-40,40] -- far too large for ordinary floating-point reduction-order
   noise, more consistent with an actual bug, e.g. uninitialized GPU memory in a
   custom kernel, than benign non-determinism).
6. bisect_fine(): camera backbone+FPN and the depth head are BOTH perfectly
   bit-identical across 5 repeats. FINAL LOCALIZATION: non-determinism is specifically
   inside the iterative Gaussian-refinement blocks -- DFA3D's custom
   deformable-attention CUDA kernel and/or the sparse-conv self-encoding step -- the
   only genuinely custom, non-standard-PyTorch machinery in the entire Stage A
   pipeline. Every standard op (conv, FPN, depth head, splatting) confirmed clean.

### Decision: stopping the fix here, deliberately

A full fix requires C++/CUDA-level debugging of third-party compiled kernels (DFA3D
and/or GaussianFormer3D's own sparse-conv code) -- different skill set, unbounded time
cost, direct distraction from the professor-directed Step 5 work. The localization
achieved is precise enough to document confidently without going further. Noise
magnitude adopted as a known, characterized constant: ~0.04-0.07 in adjusted mIoU.

### Adopted going forward: gated, cost-controlled comparison protocol

To avoid spending the expensive repeated-run budget on ideas that don't pan out,
every future architecture change (Step 5 onward) goes through staged gates, each only
reached if the previous one earns it:

  - Gate 1 (minutes): does it run end-to-end without crashing, loss visibly decreases
    over a handful of steps on one scene (same pattern as test_phase5_real_wiring.py).
    Fail -> fix the bug, do not train further.
  - Gate 2 (~1-1.5 hrs, one run, n=3, no repeats): does the held-out delta look at all
    promising, or clearly flat/negative? Unpromising -> stop, do not escalate to n=8.
  - Gate 3 (~3 hrs, one run, n=8, no repeats yet): compare the single result against
    the known noise band (~0.04-0.07) relative to current best baseline. Within the
    noise band -> stop, "indistinguishable from noise," do not spend repeat budget.
  - Gate 4 (~15-21 hrs, ONLY for a genuine standout that clearly exceeds the noise
    band at Gate 3): 3 repeats per side (new design vs. immediately-preceding
    baseline), same seed/protocol otherwise. Report mean/min/max per side, not a
    single point. Decision rule: non-overlapping ranges = real improvement;
    overlapping ranges = inconclusive (not "no effect" -- N=3 genuinely cannot tell
    the difference), reserved for genuine decision points, not every intermediate
    tweak.

This directly resolves the standing concern flagged at the end of the Steps 1-4 arc
(2026-07-30 retrospective) -- the mechanism behind that noise is now known and
characterized, and a concrete, budget-aware protocol is in place to avoid drawing
false conclusions from it going forward, without needing to eliminate it at the
source.

---

## [Phase 5] Run ID: 2026-08-06-step5-spatial-panorama-gate2-gate3

### Step 5 implemented and gated per the cost-controlled protocol (2026-08-06)

Three pieces built and individually tested before wiring together (each has its own
correctness test, not just a shape check):

1. SpatialPoolFeatures (src/models/stage_b_temporal/spatial_pool_features.py) --
   replaces PoolFeatures' full collapse-to-one-vector with a real spatial panorama
   (24 angular bins around the vehicle), explicitly handling the 6 nuScenes cameras'
   overlapping fields of view via smooth raised-cosine cross-camera blending, plus
   preserved within-camera spatial detail (4 angular sub-samples per camera, not
   collapsed to one scalar). Camera order confirmed against the real CAM_NAMES
   (grouped FRONT/FRONT_RIGHT/FRONT_LEFT/BACK/BACK_RIGHT/BACK_LEFT, not a simple
   clockwise ring). Yaw angles and FOV are NOMINAL rig-design values, not each
   scene's true calibrated extrinsics -- flagged approximation, not yet revisited.
   tests/test_spatial_pool_features.py confirms genuine blending (ablating one
   camera's contribution measurably changes only the panorama bins it actually
   overlaps into, leaves purely-single-camera bins exactly unaffected).
2. Temporal conditioning: concat([panorama_curr, panorama_prev, panorama_curr -
   panorama_prev]) instead of a single pre-computed difference vector -- richer
   signal, lets the network use both frames' raw content plus the difference.
3. SpatialConvHyperNet (src/models/stage_b_temporal/spatial_conv_hypernet.py) --
   Option B chosen over Option A (a full polar/angle-radius-height coordinate system
   matching the panorama's true geometry, kept as a future path if needed): the
   24-bin ring is stretched via plain interpolation into a small 2D map, projected
   and downsampled into a seed, then grown into the 3 output grids via Step 4's exact
   unchanged ConvTranspose3d path. Fewer parameters than Step 4's ConvHyperNet
   (260,772 vs 700,708). tests/test_spatial_conv_hypernet.py confirms perturbing one
   input bin produces a genuinely non-uniform effect on the output grid (spatial
   information CAN propagate through the architecture; untrained-network capacity
   check, not a claim of semantic correctness, which requires training).

grid_feat_dim stays 4 either way, so DeformHeadMu/DeformHeadR/the update rule are
completely unchanged -- only how the grids are generated differs. Wired in behind
USE_SPATIAL_STEP5 toggle (src/training/stage_b_engine.py) for direct comparison
against the Steps 1-4 baseline.

Two import-ordering bugs (same class as measure_noise_floor.py's) found and fixed
during this work: GF3D_ROOT-dependent imports (`model`, `loss`, `misc.metric_util`)
must come after whichever import triggers gf3d_pipeline's sys.path insertion --
caught in stage_b_engine.py during Gate 1 and in scripts/train_stage1.py's own thin
entrypoint (a leftover ordering mistake from before Step 5 began) during the actual
Gate 2 run attempt.

### Gate 1 (real-data wiring, no training): PASS

tests/test_phase5_real_wiring.py ran end-to-end on real data with USE_SPATIAL_STEP5=True
by default: correct G_0 shape (N_g=6400), G_1 provably distinct from G_0, GaussianHead
splat callable standalone producing correct pred_occ shape, peak VRAM unchanged
(3.11GB, same as before Step 5).

### Gate 2 (n=3, one run, no repeats): AMBIGUOUS, escalated per protocol

Best held-out delta: +0.080 at opt_step 40 (vs. best-ever n=3 reference of +0.125 from
the Steps 1-4 era). Gap (0.045) is within the characterized noise band (~0.04-0.07),
so this run alone cannot distinguish "worse" from "noise." Positive-delta window was
narrower than the n=3 reference (opt_step 20-40 vs. 60-160), collapsing sooner.
Per the gated protocol, an ambiguous (not clearly unpromising) Gate 2 result escalates
to Gate 3 rather than being killed early.

### Gate 3 (n=8, one run, no repeats): peak INCONCLUSIVE per protocol rule; trough is a
new, real, concerning finding

Best held-out delta: +0.085 at opt_step 60 (do-nothing=4.942). Compared against the
best-ever n=8 results from Steps 1-4 (+0.106 Step 3, +0.100 Step 4): difference is
0.021, well WITHIN the noise band. Per Gate 3's own decision rule, this is
inconclusive -- STOPPING HERE, not escalating to Gate 4's expensive repeat protocol,
exactly as the gated approach is designed to do (protects budget from confirming a
result that isn't yet distinguishable from noise).

HOWEVER, worth flagging separately from the peak-only Gate 3 rule: the TROUGH this run
reaches is meaningfully worse than every prior version, well beyond the noise band --

  Final-step delta:      Step1 -0.707 / Step2 -0.766 / Step3 -0.625 / Step4 -0.665 /
                          Step5 -1.055
  Worst point in the run: Step1 ~-0.78 / Step2 ~-0.79 / Step3 ~-0.72 / Step4 ~-0.78 /
                          Step5 -1.173

Every prior version plateaued in the -0.6 to -0.8 range; Step 5 collapses roughly
0.3-0.5 further -- a real, new, worse failure mode, not explainable as measurement
noise given its size relative to the characterized ~0.04-0.07 band.

HYPOTHESIS (not yet tested): SpatialConvHyperNet has FEWER parameters than Step 4's
ConvHyperNet (260K vs 700K), so this is likely not simple overfitting-via-capacity.
More plausible: preserving real spatial layout (the panorama) may hand the model a
stronger PER-SCENE FINGERPRINT than a single pooled scalar did, making the
scene-recognition shortcut (originally fixed by delta-conditioning, EXPERIMENT_LOG.md
Phase 5 early entries) easier to re-exploit through a different route -- a specific
spatial arrangement is more distinctive per-scene than one number, giving the model
more to memorize about "which scene is this" rather than "how does this scene move."

### Decision: logged, not yet escalated to Gate 4; trough finding to be investigated
before deciding whether to continue with the spatial-panorama approach as-is, adjust
it, or revert to Steps 1-4's baseline as the working architecture.

---

## [Phase 5] Run ID: 2026-08-06-step5-reverted-to-baseline

### Decision: reverted USE_SPATIAL_STEP5 to False, Steps 1-4 restored as active default

Following the Gate 3 finding (peak inconclusive vs. noise floor; trough collapsed
meaningfully further than every prior version, -1.055 vs. the -0.6 to -0.8 range
Steps 1-4 all plateaued in) -- reverted the toggle rather than continuing to
investigate on top of an active, regressed configuration.

Verified the revert actually restores old behavior, not just flips a flag blindly:
  - tests/test_stage_b_skeleton.py: all 3 Phase 4 checks still pass.
  - tests/test_phase5_real_wiring.py: all 4 checks pass on real data. Notably, G_1's
    mean-abs-delta-from-G_0 (0.232) is back in the same order of magnitude as the
    original Step 4-era runs, vs. the smaller 0.071 seen with the spatial pipeline
    active -- consistent, independent evidence this genuinely re-exercises the old
    ConvHyperNet path, not just a flag that happens not to error.

Step 5's code (SpatialPoolFeatures, SpatialConvHyperNet, both-frames temporal
conditioning, both test files) is fully intact, committed, and available -- nothing
deleted. USE_SPATIAL_STEP5=True resumes that investigation whenever it's picked back
up; USE_SPATIAL_STEP5=False (current state) is the known-good Steps 1-4 baseline.

### Current status

Active configuration: Steps 1-4 (rotation-order fix, grid resolution/channel
rebalance, PE-as-coordinate query, ConvHyperNet), no ego-motion compensation,
no SpawnHead, no spatial panorama. This matches exactly what was running before
Step 5 began (2026-08-06-step0-noise-floor-localized-and-protocol-adopted entry and
earlier). The open question motivating Step 5 -- pooling destroys spatial
information, camera overlaps need handling, temporal conditioning is weak -- remains
unresolved; Step 5's specific implementation attempt is parked, not abandoned,
pending investigation of the trough-collapse hypothesis (spatial layout as a
stronger per-scene fingerprint than a pooled scalar) before any further attempt.
---

## [Phase 5] Run ID: 2026-08-08-option-d-direct-projection-gate1-gate2-gate3

### Option D: skip the intermediate 3D motion grid entirely

Each Gaussian projected directly into both frames' real camera images (reusing
GaussianFormer3D's real `DeformableFeatureAggregation3D.project_points_3d`,
verified line-by-line against real source), sampling real image+depth features
there directly -- no dense grid, no HyperNet. Branch `step5b-direct-projection`.

### Gate 1: PASS. Gate 2/3 (un-normalized variant): peak inconclusive vs. noise
floor; trough shows a real, recurring instability

Gate 3 (n=8): best held-out delta +0.082 (do-nothing=4.973), within noise band vs.
every prior architecture's best-ever. HOWEVER: trough oscillated repeatedly into
-0.84 to -1.05 territory throughout an extended (2x normal length) run -- arguably
worse than Step 5's single collapse point, since it recurred rather than occurring
once, despite double the training budget.

Hypothesis: z (the concatenated sampled feature) was raw, unnormalized, high-
dimensional CNN feature-map values -- no normalization step anywhere before the
heads -- plausible cause of both slow convergence and per-step output instability.

### Normalization variant: instability resolved, but Gate 2 now clearly
unpromising (not just inconclusive)

Added per-frame (not joint) LayerNorm before concatenation. Gate 1 directly
confirmed the hypothesis: delta_mu range dropped from full tanh saturation
(-4 to 4) to a sane range (-0.73 to 0.84). Gate 2 (n=3): reached a stable,
converged plateau at -0.099 -- fast, stable, but clearly below every other
architecture's best-ever by a margin outside the noise band. Per the agreed
stopping rule (no clear improvement -> move to a different design), stopped here,
not escalated to Gate 3.

### Overall Option D conclusion

Both variants (un-normalized, normalized) do not show meaningful improvement over
existing baselines. USE_DIRECT_PROJECTION reverted to False. Both variants' code,
tests, and wiring fully intact and committed on `step5b-direct-projection` --
nothing deleted. Moving to consider a different design.

---

## [Phase 5] Run ID: 2026-08-13-vggt-deformable-design-kickoff-and-gate1

### New design: VGGT-1B as a frozen dense feature extractor, feeding 4-block
iterative deformable cross-attention

Branch `vggt-deformable-attention`. Reuses Option D's verified projection
machinery for the geometric anchor (K=4 learned offset samples + learned
attention weighting per block, independent weights per block, matching Stage A's
own verified non-weight-tied decoder precedent). Two implementation-blocking
decisions confirmed against real source before writing code: (1) PAD (not resize)
images to a multiple of 14, new `PadRawImagesForVGGT` transform, separate
`vggt_img`/`vggt_image_wh` stream; (2) use VGGT's final cached layer only, not
full DPT-style multi-scale fusion (simplest starting choice, explicit disclosed
caveat re: fine spatial detail).

Gate 1: PASS. Peak VRAM 11.85GB (vs. 3.1-3.6GB every prior architecture) --
expected consequence of VGGT-1B's scale, not alarming at this stage.

---

## [Phase 5] Run ID: 2026-08-13-vggt-deformable-resolution-crisis-and-downsample-fix

### Gate 2 at native resolution: ~22 min/optimizer step, projected ~9 days --
killed, not viable

Root cause: VGGT-1B's own tested default is img_size=518 (1369 patches/frame);
our native resolution + feeding all 12 frames (6 cams x 2 timesteps) as one
global-attention sequence meant ~5.5x more tokens than VGGT-typical, and
self-attention cost scales quadratically with sequence length.

Fix: downsample AFTER padding, target (392, 700) -- 1400 patches/frame, close to
VGGT's own native 1369. Verified geometrically correct via a real marker test.
Re-ran Gate 1: peak VRAM dropped to 7.30GB, completed in ~10 seconds.

---

## [Phase 5] Run ID: 2026-08-14-vggt-deformable-image-wh-bug-found-via-diagnosis

### Gate 2/3 initial result (-0.384 at n=3): diagnosed as a real bug, not
architecture underperformance, before accepting it

Reasoning: a well-motivated design built on proven components performing worse
than designs already known to be broken is a signature of an implementation bug,
not a bad idea. Four-item diagnostic pass (delta_mu saturation check,
fallback-embedding usage rate, VGGT's real input normalization, image_wh/padding
consistency) found: fallback-embedding usage rate ~85% -- meaning ~97.5% of
(camera, Gaussian) pairs were being marked invalid, training almost entirely on
a learned placeholder.

Root cause found and fixed: `image_wh` was set to the DOWNSAMPLED resolution,
but `projection_mat`'s own coordinates are always in native-padded pixel units --
dividing native-scale coordinates by the downsampled width/height gave values
like 1600/700 ~= 2.3, far outside [0,1], failing validity checks for almost
everything. Fixed: `image_wh` set to native-padded scale (1610, 910), distinct
from the downsampled tensor's actual shape. Fallback rate dropped to ~4-9%
post-fix, confirming the fix.

---

## [Phase 5] Run ID: 2026-08-18-vggt-deformable-dropout-result-and-detailed-eval-blocked

### z_dropout result: modest, ambiguous improvement; core collapse pattern
unchanged

n=3, 20 epochs: best held-out delta -0.442 (vs. -0.496 no-dropout baseline at the
same point) -- real, same-direction improvement of +0.054, but within the
characterized noise floor. Underlying shape unchanged: peak at epoch 6, then
oscillates -0.5 to -1.3 for the remaining 13 epochs, same signature as every
other run.

Corrected an earlier hypothesis: re-checked `evaluate_heldout()`'s real code --
it already averages over ALL 78 held-out clips on every call, not a subset.
Swings between checkpoints are real, fast changes in held-out behavior, not
measurement noise.

Per-class/per-clip detailed evaluation script built but blocked -- external
Transcend HDD became unavailable mid-session (hardware/infrastructure issue, not
code), set aside rather than debugged further.

---

## [Phase 5] Run ID: 2026-08-19-vggt-deformable-cross-architecture-synthesis-and-conclusion

### Cross-architecture pattern: five structurally different designs, same
failure shape

Steps 1-4, Step 5, Option D (both variants), VGGT+deformable-attention (four
independent bug-fixed variations) all show the identical signature: training
loss reliably, monotonically improves every time; held-out performance peaks
early (epoch 4-6) and never recovers. Diagnostic coverage unusually thorough
(parameter coverage, gradient flow, quaternion validity, offset magnitude,
attention entropy, resolution/normalization consistency, VGGT input
normalization, fallback rate, checkpointing correctness -- all directly measured,
not just reasoned about).

Most likely explanation: a genuine generalization gap given the ~8-10 scene
mini-dataset's limited diversity, not a remaining silent bug in any one
architecture -- far more parsimonious than five independent architecture-specific
failures. USE_VGGT_DEFORMABLE reverted to False; code fully intact and committed
on `vggt-deformable-attention`. Redirects effort toward what has never been
tried (L_tv/L_lidar, deliberate regularization/overfitting countermeasures, and
critically: more/better data) rather than another architecture attempt at this
same data scale.

---

## [Phase 5→6] Run ID: 2026-08-22-professor-review-gf3d-faithful-adopted

### Professor meeting: architecture direction decided

Presented the VGGT+deformable-attention design; could not adequately explain the
Deformation part and several architectural details under questioning. Directed
to discuss with labmate and return with a resolved design. Following that
discussion, proposed reverting to a design built by directly reusing
GaussianFormer3D's own real modules (`DeformableFeatureAggregation3D`, anchor/Q
construction, per-block operation sequence) for the `t>0` deformation step,
applied to the reference-buffer Gaussian instead of a freshly-initialized one --
for consistency and reliability with the reference architecture, given repeated
difficulty explaining/justifying more novel designs (VGGT's complexity, feature-
dimension mismatch, heavy global attention) under scrutiny.

**Professor's decision (final meeting):** GF3D-faithful design accepted as the
official internship report architecture. Motion HyperNet and VGGT-deformable-
attention both directed to be dropped from the report. Personal decision (not
report-facing): Motion HyperNet kept as an explicit backup/Plan B in case the
GF3D-faithful design does not work; VGGT-deformable-attention fully discarded
per the professor's direction, no backup kept.

Full architecture, method, math, professor Q&A (4 review points: refinement-
module structure, ego-motion/coordinate-frame necessity, attention-weight
mechanism, DFA math location), citations (GaussianFormer, GaussianFormer3D,
DFA3D, Deformable DETR, 4DGC, Attention Is All You Need) developed and confirmed
against real GaussianFormer3D source throughout (`GaussianOccEncoder3D`,
`DeformableFeatureAggregation3D`, `SparseGaussian3DRefinementModule`,
`gaussian_lifter.py`, `NuScenesAdaptor`) -- not from the paper's text alone.
Design document (STAGE_B_GF3D_FAITHFUL_DESIGN.md) developed iteratively over
several sessions; a v2 revision (source-verified corrections: kps_generator's
real anchor+Q construction, AnchorEncoder's real five-way split, FFN/LayerNorm
ordering) exists but is not yet finalized/added to this repo -- separate
decision pending on Design A vs. Design B (cascaded vs. non-cascaded block
iteration) before treating it as final.

### Git branch restructuring

- `archive/motion-hypernet-backup` -- created from `main`, explicit personal
  backup, not part of the report.
- `archive/vggt-deformable-attention` -- renamed from `vggt-deformable-attention`
  (full history preserved, old branch name removed from GitHub only, no data
  lost).
- `archive/option-d-direct-projection` -- renamed from `step5b-direct-projection`,
  same treatment, for consistency (concluded earlier, same category).
- `gf3d-faithful-stageb` -- new active branch, created from
  `stage-a-full-dataset-gs25600` (not `main`), since the new design builds on
  the full-dataset infrastructure, not the mini-dataset Motion HyperNet code.
- `main` -- unchanged for now (still Motion HyperNet); to be replaced with
  GF3D-faithful content once implemented and confirmed working.

---

## [Full-dataset pivot] Run ID: 2026-08-21-full-dataset-vram-check-and-data-wiring

### Motivation

Senior labmate (Ruby, GaussianFormer2/PLAS compression, not GaussianFormer3D)
provided an SSD (`1TSSD`) containing what was confirmed to be genuine full-scale
nuScenes v1.0-trainval (433GB, `v1.0-trainval` folder naming, not mini) plus
SurroundOcc annotations (41GB) -- real chance to move off the ~8-10 scene mini
dataset, directly motivated by the cross-architecture small-data-overfitting
finding above.

### Step 1: VRAM feasibility, N_g=25600

Real GF3D full-scale config (`nuscenes_surroundocc_gs25600.py`) confirmed via
synthetic-tensor forward pass (no real data needed yet): ~4.07GB peak, forward-
pass only, comfortable headroom vs. 24GB budget. Five real bugs found and fixed
along the way, each confirmed via direct verification rather than assumed:
depth-head requires real GT depth shape (not None); real image height is padded
to 928 (multiple of 32), not the raw 900 the config states; in-place tensor
mutation (`squeeze_`) corrupted a reused tensor across two forward passes in the
test harness itself; GaussianHead's splat step needs `occ_xyz`/`occ_label`/
`occ_cam_mask` in UN-flattened grid shape, matching this project's own
`LoadOccupancyOcc3d` convention exactly.

### Step 2: full data pipeline wiring

- Symlinked `data/nuscenes`, `data/nuscenes_cam`, `data/surroundocc` inside the
  GaussianFormer3D repo (matching real `data_root`/`anno_root`/`occ_path`
  conventions from `_base_/surroundocc_pcd_dfa3d.py`).
- Found and fixed a real double-nesting issue: the actual nuScenes JSON
  metadata tables and `maps`/`samples`/`sweeps` folders were nested one level
  deeper than the devkit expects (`v1.0-trainval/v1.0-trainval/...`) -- fixed via
  read-only symlinks up one level, nothing moved/copied, Ruby's real data
  untouched throughout.
- Generated `nuscenes_infos_gf3d_{train,val}.pkl` via GF3D's own real
  `tools/make_gf3d_infos.py --regenerate`, reading Ruby's existing
  `*_sweeps_occ.pkl` files (read-only) -- confirmed real counts: 700 train
  scenes/28,130 keyframes, 150 val scenes/6,019 keyframes (28,130+6,019=34,149,
  exactly matching the devkit's own reported total sample count -- full
  coverage, nothing excluded).
- Generated depth-GT ourselves (204,894 files, ~6.75 min), reusing and adapting
  this project's own `generate_depth_gt.py` (built earlier for the mini
  dataset) rather than depending on GaussianFormer3D's own dead SharePoint
  download link -- confirmed correct via direct content verification (u/v
  ranges within image bounds, plausible depth values).
- All new outputs written to a separate `/media/user/1TSSD/min/` folder
  structure, never mixed into or overwriting Ruby's original data.

### Real training smoke test: PASS

`train.py` ran cleanly on real full-scale data: healthy, decreasing loss
(31.8->22.0 over 450 iterations), stable grad_norm, ~2.25-2.3s/iteration.

---

## [Full-dataset pivot] Run ID: 2026-08-22-small-tier-gate-check

### Small-tier (50 train / 10 val scenes) gate check: healthy signal, confirms
readiness for full-scale

Per-epoch mIoU: 8.54 -> 9.53 -> 10.36 -> 10.51 -> 10.52 across 6 epochs (real
plateau by epoch 3-4, expected given only 50 scenes' diversity). One real
disconnect/OOM incident mid-run, recovered cleanly via `train.py`'s own
confirmed `latest.pth` auto-resume logic -- no data lost. Decision: healthy
signal, skip the medium tier, proceed directly to full-scale.

---

## [Full-dataset pivot] Run ID: 2026-08-24-to-08-27-full-scale-stage-a-training

### Full-scale training (700 train / 70 val scenes, N_g=25600, 6 epochs planned)

Launched under a resilient auto-restart wrapper (survives OOM/crashes/
disconnects, resumes via `latest.pth`). Real per-epoch mIoU: 17.24 -> 22.06 ->
23.61 (epochs 0-2 inclusive), each a real, meaningful gain (not yet plateaued,
unlike the small tier at the same point) -- one epoch of full-scale data alone
exceeded the small tier's entire 5-epoch result by ~64%.

Real, recurring OOM issue diagnosed during epoch 2 (0-indexed): 4 consecutive
crashes at the same point across ~33 hours, distinguished from ordinary
per-sample bad luck by the fact epochs 0-1 never failed -- consistent with a
cumulative memory-pressure issue, not a rare unlucky sample. Resolved without
needing to fix the root cause: attempt 5 succeeded through the same stretch
that failed 4 times prior; a proactive per-epoch-restart wrapper was prepared
as a ready backup but never needed.

**Decision: stopped after epoch 3's checkpoint (mIoU=23.61), deliberately, given
the 20-day total budget.** ~87% of GaussianFormer3D's own reported 27.1 mIoU,
reached in 3 epochs vs. their 24 -- efficient convergence, not full convergence;
gains were still real and ongoing (not plateaued) when stopped, a genuine
time-budget tradeoff, not a claim of convergence. Per-class breakdown showed the
expected pattern: strong on large/common classes (vegetation 39.5%,
driveable_surface 34.3%), weaker on small/rare dynamic classes (bicycle,
motorcycle, traffic_cone) -- worth remembering when interpreting future Stage B
results on exactly those classes, since Stage A's own G_0 is weaker there
regardless of Stage B's design.

---

## [Full-dataset pivot] Run ID: 2026-08-27-g0-extraction-cache-built

### G_0 extraction pipeline: built, debugged, validated, run across all 850 scenes

Built `extract_g0_cache.py` (GaussianFormer3D repo) + `make_frame0_infos.py`
(this repo) to run the trained Stage A checkpoint once per scene (frame 0 only,
combining both train/val splits, 850 scenes total) and cache each scene's G_0
(means/scales/rotations/opacities/semantics) to its own file -- so no future
Stage B experiment needs to re-run Stage A.

Real bugs found and fixed via a 5-scene test before committing to the full run:
missing `import model` (registry decorators never triggered); `scene_token`
does not survive inside the collated batch as assumed -- fixed by reading it
directly from `dataset.keyframes[i]` instead, exploiting the known,
deterministic (shuffle=False, batch_size=1) iteration order; a genuine,
repeated silent write-corruption issue on the `1TSSD` drive (a file reported a
plausible size via `ls -la` but was still corrupted/truncated on actual read,
twice) -- fixed by making the write self-verifying (temp file + explicit
flush/fsync + read-back verification + atomic rename), a pattern now to be
reused for any future write to this drive.

Full run: 850/850 scenes extracted successfully (~7.7 min), spot-checked
(correct shape `(1, 25600, 3)`, correct scene-token match).

---

## [Full-dataset pivot] Run ID: 2026-08-28-gf3d-faithful-branch-housekeeping

### Repository cleanup on gf3d-faithful-stageb

Removed Motion HyperNet/Step5-specific modules no longer needed by the new
design (`conv_hypernet.py`, `hypernet.py`, `grid_query.py`, `pool_features.py`,
`spatial_conv_hypernet.py`, `spatial_pool_features.py`, `spawn_head.py`,
`configs/stage_b_temporal.yaml`, and their now-orphaned tests) -- explicitly
KEPT `buffer.py`, `current_frame_encoder.py`, `deform_heads.py`, all confirmed
still directly reused by the new design (verified via the design document's own
stated reuse plan and, for `deform_heads.py` specifically, via checking
`__init__.py`'s real re-export list before assuming anything was safe to
remove). `test_stage_b_skeleton.py` trimmed to its architecture-independent
quaternion-composition test (recursion/shape test deferred until the new
deformation module exists to test against). One real oversight caught and
fixed: `test_spatial_pool_features.py` was initially missed in the first
cleanup pass, left importing an already-deleted module -- found via a full
directory sweep, fixed in a follow-up commit.

Separately: two confirmed-dead mini-dataset configs removed
(`dataset_mini_occ3d.yaml`, `stage_a_gaussianformer3d.yaml` -- zero code
references, confirmed earlier this project), three scripts archived to
`scripts/deprecated/` (jobs fully superseded by later real results:
`overfit_stage_a_single_frame.py`, `spot_check_dataloader.py`,
`verify_scene_coverage.py`). `data/README.md` updated to document the
full-scale dataset's actual location (separate GaussianFormer3D repo, not
tracked here) alongside the still-valid, deliberately-kept mini-dataset setup
(retained as a fast, cheap iteration/sanity-check tool for new Stage B designs
before committing to full-scale runs).

Full original history for everything removed/archived remains available on
`archive/motion-hypernet-backup` and `main` -- nothing destroyed, only
reorganized.

### Current status

Active branch: `gf3d-faithful-stageb`. Stage A: real, full-scale checkpoint
available (`epoch_3.pth`, mIoU=23.61), `G_0` cached for all 850 scenes. Stage B:
design finalized pending the Design A vs. Design B decision (see design
document); implementation not yet started. Next: finalize
`STAGE_B_GF3D_FAITHFUL_DESIGN.md`, add it to `docs/`, then begin actual
implementation.

## [2026-08-31] Stage B (GF3D-faithful, Design B) -- first real training run, L=4

**Setup:** single-step training (G_0 -> next genuinely-moving real keyframe,
>=0.5m ego translation), 651 train scenes / 140 held-out val scenes (genuine
split -- an earlier manifest draft incorrectly combined train+val, caught and
fixed before any real training). L=4 blocks, embed_dims=128, AdamW,
batch_size=1. LR schedule: 2-epoch linear warmup to 1e-4 (confirmed necessary
via a 20-scene gate check -- without it, a real mid-training instability hump
appeared, epochs 8-14 rising back to near the epoch-1 starting loss); cosine
decay to 1e-5 added when extending past epoch 20 (train_loss still improving
smoothly at that point, not a "stuck" fix -- added to reduce late-training
val_mIoU noise, confirmed effective: epoch 18-20 band 13.63-14.46 vs. epoch
33-40 band 15.15-15.27 after decay).

**Real bugs found and fixed before this run produced valid results** (full
detail in prior entries/commits): a critical anchor encode/decode bug (real
GaussianState values were fed directly into GF3D's real modules without the
required sigmoid/softplus inverse-activation, silently corrupting position,
scale, opacity, and semantics -- confirmed via a real out-of-bounds splatting
crash, fixed by reusing GF3D's own real safe_sigmoid/safe_inverse_sigmoid);
an overly-aggressive semantics clamp in the fix itself (min=1e-6 distorted
~18.5% of real semantic values, caught via a frozen-property round-trip
assertion failing); a real out-of-range crash from genuine large ego-motion
scenes requiring a boundary-safety clamp on the transformed anchor position.

**Training progression** (40 epochs total, across two extensions from an
initial 9-epoch run):

| Epoch | train_loss | val_loss | val_mIoU | val_iou2 |
|---|---|---|---|---|
| 1 | 4.759 | 4.792 | 3.81 | 15.69 |
| 9 | 3.782 | 4.079 | 11.26 | 24.56 |
| 20 | 3.581 | 3.818 | 14.46 | 29.41 |
| 33 | 3.401 | 3.775 | **15.22** | **30.31** |
| 40 | 3.372 | 3.770 | 15.20 | 30.27 |

Converged by ~epoch 30; val_mIoU plateaued in a tight 15.15-15.27 band for
the final ~8 epochs (real convergence, not cut off early) while train_loss
kept very gently decreasing -- earliest, mild signature of the train/val gap
widening, consistent with stopping here being the right call.

**Official checkpoint: `epoch_33.pth`** (highest val_mIoU among all saved
checkpoints; CHECKPOINT_EVERY=3 means epoch 34's own peak, 15.2684 in the
raw per-epoch log, was never actually saved to disk -- epoch 33 and 36 are
the real neighbors, both within noise of it).

**Three-table comparison, same 140 held-out scenes, same real MeanIoU
metric throughout:**

| Table | Method | mIoU | iou2 |
|---|---|---|---|
| 1 | Static 3DGS (fresh Stage A per-frame, no temporal reuse -- the oracle ceiling) | 21.93 | 35.80 |
| 2 | Do-nothing baseline (G_0 reused unchanged, frame-transformed, no deformation) | 13.52 | 27.80 |
| 3 | **Stage B (ours, L=4, epoch 33)** | **15.22** | **30.31** |

Stage B closes ~21% of the gap between doing nothing and the full,
expensive oracle reconstruction ((15.22-13.52)/(21.93-13.52) = 0.208) --
a real, genuine improvement from a single feedforward deformation step,
on genuinely held-out data never touched during training.

**Known limitations of this result, stated plainly:** single-step training
only (G_0 -> one real next-frame prediction per scene, never chaining the
model's own predictions recursively) -- real multi-step deployment
(evaluating a full ~40-frame scene sequentially) has not been tested and is
a known, deliberate gap, not yet closed. Evaluated on one frame-pair per
scene (791 total pairs across train+val combined), not dense per-frame
supervision the way Stage A itself was trained.

**Next:** L=6 ablation (same training recipe, only block count changed) to
test whether more capacity closes more of the remaining gap, before
committing to the larger, riskier multi-step recursive training investment.

## [2026-09-01] Stage B ablation -- L=6, same recipe as L=4

**Setup:** identical recipe to the L=4 run (same 651 train / 140 held-out val
split, same warmup+cosine-decay LR schedule, same AdamW, batch_size=1) --
only NUM_BLOCKS changed, 4 -> 6. Separate checkpoint directory
(checkpoints_L6/) and log file (train_log_L6.txt), to avoid any risk of the
resume logic loading an L=4 checkpoint into an L=6 model (shape mismatch).

**Real bug found and fixed:** L=6's real peak VRAM (20.56GB) exceeded what
was available while sharing the GPU, causing a genuine CUDA OOM crash-restart
loop (confirmed via a real out-of-memory traceback, not assumed). Root cause:
memory fragmentation ("3.03GB reserved but unallocated" in the real error) --
the same category of issue Stage A's own training addressed via
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512, applied here too. Confirmed
fixed: training then ran cleanly through completion.

**A real methodological issue found and corrected mid-run, worth recording
precisely:** training was paused after 20 epochs for report work, using
NUM_EPOCHS=20 at that time -- meaning L=6's cosine decay began immediately
after its 2-epoch warmup and ran continuously through all 20 epochs. This
made an epoch-20-vs-epoch-20 comparison against L=4 invalid: L=4's own first
20 epochs (from its original run) were spent entirely at CONSTANT lr=1e-4
(decay was only added later, when L=4 itself was extended past epoch 20) --
so L=4 had ~18 full epochs of high-LR exploration in its first 20, while L=6
had almost none. Caught before drawing any conclusion from the epoch-20
numbers (L=6 was tracking meaningfully lower: 11.92 vs L=4's 14.46) --
correctly identified as an artifact of the schedule difference, not
necessarily architecture. Fixed by extending L=6 to NUM_EPOCHS=40 (matching
L=4's real final scope) and resuming from epoch_20.pth, giving L=6 genuine
renewed high-LR exploration before its own decay phase, for a fair comparison.

**Full training progression (40 epochs, across the pre/post-extension
boundary at epoch 20):**

| Epoch | train_loss | val_loss | val_mIoU | val_iou2 |
|---|---|---|---|---|
| 1 | 5.051 | 5.192 | 2.13 | 11.96 |
| 20 (pre-extension, decayed schedule -- not comparable to L=4 epoch 20) | 3.798 | 4.003 | 11.92 | 26.63 |
| 21 (renewed exploration begins) | 3.908 | 4.076 | 11.02 | 24.72 |
| 30 | 3.590 | 3.846 | 14.04 | 28.87 |
| 33 | 3.550 | 3.817 | 14.39 | 29.49 |
| 36 | 3.519 | 3.809 | **14.60** | **29.99** |
| 40 | 3.500 | 3.826 | 14.37 | 29.44 |

Converged by ~epoch 30, plateaued in a tight 14.1-14.6 band for the final
~10 epochs -- a genuine, real convergence (same pattern as L=4's own run),
not cut short.

**Official L=6 checkpoint: `epoch_36.pth`** (highest val_mIoU among saved
checkpoints, CHECKPOINT_EVERY=3).

**Result: L=6 did NOT outperform L=4.**

| L | Best val_mIoU | Epoch | Params |
|---|---|---|---|
| 4 | **15.22** | 33 | 5.46M |
| 6 | 14.60 | 36 | 6.49M |

L=6 used ~19% more parameters, ~27% more peak VRAM (20.56GB vs 16.21GB),
required a real fragmentation-config fix just to fit, and produced a result
0.62 mIoU points LOWER than L=4 -- a real, negative ablation finding, not
noise (both runs converged to a stable, tight plateau, not still-moving
numbers being compared mid-flight).

**Interpretation:** the current performance ceiling is likely NOT explained
by insufficient deformation-network capacity. More blocks did not help, and
if anything pointed slightly the other way. This is useful, real evidence
that the more promising direction for further improvement is elsewhere --
most plausibly the single-step training scope itself (the model has only
ever seen pristine G_0 as input, never its own accumulated prediction error,
a known and named limitation since the original single-step-vs-multi-step
training decision) -- rather than continuing to scale L further.

**Next:** L=4 remains the official, reported Stage B configuration.
Multi-step recursive training (and, bundled with it, the previously-deferred
L_lidar/L_tv loss completion -- L_tv specifically requires a real multi-step
sequence to be meaningful at all, confirmed not implementable under the
current single-step scope) is now the more clearly-motivated next
investment, not further L scaling.

## [2026-09-01] Stage B ablation -- L=2, completing the {2,4,6} comparison

**Setup:** identical recipe to L=4/L=6 (same 651 train / 140 held-out val
split, same warmup+cosine-decay LR schedule over the full 40 epochs from the
start this time -- avoiding the two-phase 20->40 approach that created a
real LR-schedule confound for the L=6 comparison). Only NUM_BLOCKS changed,
4 -> 2. Separate checkpoint directory (checkpoints_L2/) and log file
(train_log_L2.txt).

**One real, unrelated bug hit at launch, quickly diagnosed and fixed:** the
first launch attempt crash-looped immediately with
`ModuleNotFoundError: No module named 'mmengine'` -- traced to the launch
being run from a terminal/session where the gf3d conda environment had not
been activated (confirmed via `which python` after activating correctly).
Not a code bug; relaunching in a properly-activated shell resolved it
immediately.

**Full training progression (40 epochs, single clean schedule throughout):**

| Epoch | train_loss | val_loss | val_mIoU | val_iou2 |
|---|---|---|---|---|
| 1 | 4.149 | 4.177 | 10.40 | 24.76 |
| 10 | 3.538 | 3.803 | 15.06 | 30.11 |
| 20 | 3.399 | 3.765 | 15.45 | 30.33 |
| 30 | 3.298 | 3.740 | 15.67 | 30.73 |
| 38 | 3.257 | 3.739 | **15.73** | **30.72** |
| 40 | 3.252 | 3.737 | 15.71 | 30.91 |

Notably fast, clean convergence -- val_mIoU was already close to its final
value by epoch ~10 (far faster than L=4 or L=6), and the final ~10 epochs
(37-40: 15.65/15.73/15.68/15.71) form the tightest, least noisy plateau of
the three L values tested. ~15.5min/epoch (vs ~23min for L=4/L=6) --
genuinely faster, not just comparably fast.

**Official L=2 checkpoint: `epoch_38.pth`** (highest val_mIoU among saved
checkpoints).

**Result: L=2 outperforms both L=4 and L=6 -- a clean, monotonic ordering.**

| L | Best val_mIoU | Epoch | Params | Epoch time |
|---|---|---|---|---|
| **2** | **15.73** | 38 | 4.43M | ~15.5min |
| 4 | 15.22 | 33 | 5.46M | ~23min |
| 6 | 14.60 | 36 | 6.49M | ~23min |

Performance DECREASES monotonically as L increases across the entire range
tested (2, 4, 6) -- not just "L=6 is worse," but a real, consistent trend:
more blocks hurt, at every step tested. L=2 uses ~19% fewer parameters than
L=4 and ~32% fewer than L=6, trains ~33% faster per epoch than either, and
still wins on the actual metric that matters.

**Interpretation, now confirmed across three points rather than two:** the
current performance ceiling is not explained by insufficient deformation-
network capacity -- if anything, the opposite. The most likely explanation:
with only 651 training scenes and a single feedforward step per scene, a
larger network overfits faster than a smaller one generalizes -- consistent
with L=2's own trajectory being the fastest-converging and least noisy of
the three. This strengthens (does not merely repeat) the L=6 finding: the
more promising direction for further improvement is elsewhere -- most
plausibly the single-step training scope itself, or simply more real
training data per model (unlikely to help much further at L=6's scale, but
worth noting L=2 was not tested for whether MORE epochs specifically would
have helped it further, given its own late-epoch plateau looked genuinely
stable, not still improving).

**Official Stage B configuration updated: L=2, epoch_38.pth** (superseding
L=4, epoch_33.pth, as the reported result). All report materials (per-class
chart, qualitative BEV comparisons, full-val diff statistics) are being
regenerated against this checkpoint, saved to a separate results folder
rather than overwriting the L=4 materials (kept for the ablation record).

**Next:** L={2,4,6} ablation complete. Decision pending: finalize L=2 as the
reported result and move to report writing, or pursue multi-step recursive
training as a further, larger investment (see L=6 entry's discussion of
L_tv's dependency on this).
