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
bash ```
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