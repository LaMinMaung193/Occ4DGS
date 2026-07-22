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