# Experiment Log — Occ4DGS

Running research log. Copy the block from `docs/EXPERIMENT_LOG_TEMPLATE.md` for every run.
This file (not memory, not Slack messages to labmates) is the source of truth for paper
writing and professor check-ins.

## Summary table (update after every logged run)

| Run ID | Phase | N_g | Window | Stage | LR | Overall mIoU | Overall IoU | VRAM peak | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-07-12-env-verification | 0 | — | — | — | — | — | — | — | env verified, see entry below |

---

## Entries

## [Phase 0] Run ID: 2026-07-12-env-verification

- **Git commit:** (fill in after this commit)
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
edited 3D-deformable-attention/DFA3D/setup.py: c++14 -> c++17
(both extra_compile_args['cxx'] and ['nvcc'])
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
  1. DFA3D's `setup.py` hardcoded `-std=c++14` in both
     `extra_compile_args['cxx']` and `['nvcc']`, incompatible with torch 2.1's
     C++17-required ATen headers (`#error C++17 or later compatible compiler
     is required to use ATen`). Fixed by editing both occurrences to
     `c++17`, then `rm -rf build/ dfa3D.egg-info/` before rebuilding to clear
     stale compiled objects.
  2. `mmsegmentation` 1.2.1's top-level `__init__` eagerly imports its full
     backbone zoo (BEiT etc.), which needs `ftfy` + `regex` for CLIP-style
     tokenization even though this project doesn't use those backbones.
     Fixed with `pip install ftfy regex`.
  `unittest_DFA3D.py` took ~1hr wall-clock with no console output —
  consistent with a gradcheck-style float64 numerical verification, not a
  hang (confirmed no zombie process via `ps aux`, no OOM in `dmesg`).
- **Bugs / issues encountered & fixes:** see Observations above.
- **Decision / next step:**
  Environment fully verified. Proceeding to Phase 1 (frame index & data
  loading, `docs/IMPLEMENTATION_ROADMAP.md`). `requirements.txt` updated
  with exact working versions.

  ## [Phase 0] Run ID: 2026-07-14-pc-range-verification

- Git commit: (fill in after this commit)
- Config file(s): configs/dataset_mini_occ3d.yaml, configs/stage_a_gaussianformer3d.yaml
- Command: np.load() inspection of data/occ3d_gts/scene-0061/023c4df2.../labels.npz
- Results:
  - semantics: shape (200,200,16), dtype uint8, range 0-17 — matches configured
    pc_range=[-40,-40,-1,40,40,5.4], voxel_size=0.4m, 18 classes. No mismatch.
  - mask_lidar: shape (200,200,16), dtype uint8, binary — LiDAR visibility mask,
    not previously accounted for in configs; candidate input for L_lidar (Phase 6).
  - mask_camera: shape (200,200,16), dtype uint8, binary — camera visibility mask,
    matches configs/dataset_mini_occ3d.yaml's use_camera_visibility_mask flag.
- Observations: Occ3D-nuScenes ships semantics + two SEPARATE binary masks in one
  labels.npz, not a combined mask or an embedded special value. src/datasets/occ3d_gt.py
  (Phase 1) needs to load and return all three arrays, not just semantics.
- Decision / next step: Phase 0 fully complete (all 5 exit checklist items closed).
  Proceeding to Phase 1 (frame index & data loading).


  ## [Phase 1] Run ID: 2026-07-14-frame-index-and-gt-loader

- Git commit: (fill in after commit)
- Config file(s): configs/dataset_mini_occ3d.yaml
- Command: scripts/build_frame_index.py, scripts/spot_check_dataloader.py
- Hardware: N/A (CPU-only, dataset indexing)
- Hypothesis / what this run tests:
  Build a reliable has_gt-tagged frame index across all 10 scenes before any
  model touches the data (Phase 1, roadmap steps 1-4).
- Results:

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

- Observations:
  First pass showed a clean ALL-OR-NOTHING pattern: 6 scenes at 100% GT
  coverage, 4 scenes at exactly 0%. Root cause (confirmed by direct file
  check, not guessed): this Occ3D-nuScenes GT dump uses TWO different
  on-disk layouts depending on scene:
    (a) data/occ3d_gts/<scene>/<token>/labels.npz   (subfolder + labels.npz)
    (b) data/occ3d_gts/<scene>/<token>.npz           (flat file)
  scene-0061, 0103, 0757, 0796, 1077, 1094 use (a); scene-0553, 0655, 0916,
  1100 use (b). No correlation found with log location (boston-seaport vs
  singapore) or log file naming that would predict which convention a scene
  uses -- likely an artifact of how this GT release was assembled/merged
  from multiple download batches, not a semantic pattern to rely on.
  Fixed by checking both path candidates in load_occ3d_labels() rather than
  assuming one convention. After the fix: 100% GT coverage across all 10
  scenes (better than the old 3DGS-project "mini_train index 39 gap" would
  have suggested -- that gap does not appear to apply to this exact set of
  10 scenes/this GT release, or was specific to a different data path).
- Bugs / issues encountered & fixes: see Observations above.
- Decision / next step:
  Phase 1 exit checklist fully satisfied (all 10 scenes have full-length
  contiguous valid-GT runs, well above the 3-frame minimum needed for
  Stage 2's unroll window). experiments/phase1_frame_index.json is now the
  source of truth for all later phases -- do not recompute this differently
  elsewhere. Proceeding to Phase 2 (Stage A standalone reproduction).