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

### Option D implemented and gated per the cost-controlled protocol

DirectProjectionSampler (src/models/stage_b_temporal/direct_projection_sampler.py)
built and tested standalone (tests/test_direct_projection_sampler.py) per
OPTION_D_DESIGN.md, verified line-by-line against the real GaussianFormer3D source
(project_points_3d) before implementation, not assumed from the design doc alone.
Wired into stage_b_engine.py behind USE_DIRECT_PROJECTION, mutually exclusive with
USE_SPATIAL_STEP5. Skips the intermediate 3D motion grid entirely -- each Gaussian is
projected directly into both frames' real camera images and samples real image+depth
features there, concatenated as z=[f_prev, f_curr, f_curr-f_prev] (720-d), fed into
unchanged DeformHeadMu/DeformHeadR (in_dim switched from 24 to 720). Branch
step5b-direct-projection, commits e29bfac (module+test) / 05b66b7 (wiring).

### Gate 1 (real-data wiring, no training): PASS

tests/test_phase5_real_wiring.py ran end-to-end on real data with
USE_DIRECT_PROJECTION=True: correct G_0 shape (N_g=6400), G_1 provably distinct from
G_0 (mean abs delta 1.996), GaussianHead splat callable standalone producing correct
pred_occ shape, peak VRAM 3.11GB -- identical to the Steps 1-4/Step 5 baseline
despite per-Gaussian real-image sampling replacing pooling (no memory regression from
the architecture change).

### Gate 2 (n=3): ambiguous at default budget, escalated after a budget extension

Default 20-epoch/585-opt-step budget: best held-out delta -0.086 at opt_step 560,
never crossed positive. Unlike every prior architecture (which peaked early, opt_step
20-80, then degraded), this run was still monotonically climbing at the final eval
point -- a qualitatively different trajectory shape than the "early peak, then
collapse" pattern the gate budget was tuned around.

Extended to 40 epochs (1170 opt steps) to test whether the climb would continue: best
held-out delta improved to +0.076 at opt_step 560, crossing positive for the first
time at opt_step 420. Compared to Steps 1-4's best-ever n=3 (+0.125) and Step 5's
ambiguous n=3 (+0.080): this sits in the same ambiguous territory as Step 5, not
clearly better or worse.

Notably, once past opt_step 420, the trough looked shallower than any prior
architecture at this scale -- oscillating roughly -0.42 to +0.08 for the remainder of
the extended run, vs. Steps 1-4's -0.6 to -0.8 characteristic plateau or Step 5's
-1.055 collapse. This did NOT hold up at Gate 3 (see below).

Per protocol, an ambiguous (not clearly unpromising) Gate 2 result escalates to
Gate 3.

### Gate 3 (n=8, 40 epochs/3160 opt steps -- budget not reverted to the standard
20-epoch/~1580-opt-step reference before this run, an oversight; the run is roughly
2x the length of every prior architecture's Gate 3): peak INCONCLUSIVE per protocol
rule; trough shows a real, recurring instability, arguably worse than Step 5's

Best held-out delta: +0.082 at opt_step 1600 (do-nothing=4.973). Compared against the
best-ever n=8 results from Steps 1-4 (+0.106 Step 3, +0.100 Step 4) and Step 5's
Gate 3 (+0.085): difference from all three is 0.018-0.024, well within the ~0.04-0.07
noise band. Per Gate 3's own decision rule, this is inconclusive.

HOWEVER, distinct from Step 5's failure mode: rather than a single collapse point,
this run's trough OSCILLATES repeatedly and continuously into deep-negative territory
throughout the second half of an unusually long run, never stabilizing:

  -0.863 (1000)  -0.955 (1080)  -0.958 (1160)  -1.045 (1240)
  -0.886 (1320)  -0.840 (1400)  -1.052 (1480)  -0.836 (1560)
  -0.969 (1640)  -0.850 (1800)  -0.794 (1960)  -0.849 (2040)

These recur at roughly the same magnitude as Step 5's single flagged -1.055 collapse,
but repeatedly, all the way to the end of a budget roughly 2x longer than any prior
architecture's Gate 3 -- i.e. more training did not resolve the instability; it reads
as a per-step oscillation, not a converging trend.

HYPOTHESIS (not yet tested): unlike every prior architecture, whose z came from
query_motion_grid_pe_coordinate's bounded [-1,1] sin/cos positional encoding,
DirectProjectionSampler's z is raw, unnormalized, high-dimensional (720-d)
concatenated CNN feature-map values (image + depth), with no normalization step
anywhere in the module before DeformHeadMu/DeformHeadR. An unbounded, unnormalized,
higher-dimensional input is a plausible cause of both the slower Gate 2 convergence
and this per-step output instability -- small weight updates producing large output
swings -- though not yet confirmed.

### Decision: logged, not escalated to Gate 4; stopping per protocol, matching the
Step 5 precedent

Per the established protocol and the precedent set with Step 5 (peak inconclusive vs.
noise floor does not by itself justify continuing to Gate 4's expensive repeat
protocol), and given the trough finding here is arguably a MORE clear-cut concern
than Step 5's (recurring rather than a single collapse point, despite double the
training budget) -- stopping here. USE_DIRECT_PROJECTION reverted to False; Steps 1-4
remains the active baseline.

DirectProjectionSampler's code, test, and wiring are fully intact and committed
(branch step5b-direct-projection, commits e29bfac / 05b66b7) -- nothing deleted.
USE_DIRECT_PROJECTION=True resumes this investigation whenever picked back up.

### Next planned step (not yet started)

The core idea (real per-Gaussian geometric projection into real camera images,
replacing every approximation the grid-based approaches introduced) remains
architecturally well-motivated and is not being abandoned outright. Before moving to
a different design, plan to test the normalization hypothesis above as a targeted,
minimal variant: add a normalization step (e.g. LayerNorm) to z before
DeformHeadMu/DeformHeadR, re-run Gate 1->2->3 fresh (not a continuation of this run).
Agreed stopping rule going in: if this variant does not show clear, meaningful
improvement over this entry's numbers, move on to a different design rather than
continuing to iterate on this one.
