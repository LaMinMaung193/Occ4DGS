# Occ4DGS: Feedforward Online Dynamic 3D Gaussian Splatting for Occupancy Prediction

**A feedforward, online alternative to per-frame Gaussian reconstruction for 3D semantic occupancy prediction in autonomous driving.**

CCU Autonomous Driving Perception Lab, advised by Prof. Jui-Chiu (Rachael) Chiang.

---

![Architecture](docs/assets/architecture.jpg)

## Overview

Dense 3D semantic occupancy prediction is important for safe autonomous driving. Recent methods represent a scene as a sparse, object-centric set of 3D Gaussians rather than a dense voxel grid — but reconstruct this representation entirely from scratch on every frame, treating each frame as an independent, static scene.

**Occ4DGS** instead proposes a feedforward, online dynamic Gaussian representation: a **Static Gaussian Generation** module reconstructs a scene once, from its first frame, reusing [GaussianFormer3D](https://github.com/NVlabs/GaussianFormer3D)'s own real reconstruction pipeline unchanged. A lightweight **Dynamic Deformation** module then propagates this representation forward in time, updating it for every subsequent frame as it arrives by predicting the scene's own residual motion — without re-running full reconstruction or requiring future frames.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the complete architecture, module-by-module flow, and formal math.

## Key Results

All results evaluated on the same 140 held-out validation scenes from the official nuScenes train/val split.

| Method | mIoU ↑ | Params | Latency (ms) ↓ |
|---|---|---|---|
| Static 3DGS (fresh, per-frame reconstruction) | **25.13** | 57.61M | 548.2 |
| Do-nothing baseline (no deformation) | 16.75 | — | — |
| **Occ4DGS, Dynamic Deformation (L=2, ours)** | **18.44** | **4.43M** | **359.5** |

The Dynamic Deformation module improves meaningfully over the do-nothing baseline while requiring only a single lightweight feedforward pass per frame, using roughly **13× fewer parameters** than the full reconstruction pipeline it replaces.

See [`docs/RESULTS.md`](docs/RESULTS.md) for full per-class breakdowns, the complete efficiency benchmark, the ablation study, and qualitative comparisons across four representative scenes.

## Companion Repository

Stage A (Static Gaussian Generation) and all real training/evaluation/benchmarking scripts for Stage B run against a modified fork of GaussianFormer3D:

**[GaussianFormer3D (modified fork)](../GaussianFormer3D)** — see its own `README.md` and `occ4dgs_scripts/` directory.

## Installation

```bash
conda create -n gf3d python=3.8
conda activate gf3d
pip install -r requirements.txt
```

This project also requires a working [GaussianFormer3D](../GaussianFormer3D) installation (its own `README.md`/`docker/` cover its specific dependencies — mmdet3d, mmcv, spconv, DFA3D).

## Data and Checkpoints

See [`docs/DATA_AND_CHECKPOINTS.md`](docs/DATA_AND_CHECKPOINTS.md) for the full list of required data files and trained checkpoints, with download links and expected directory layout.

## Reproducing the Results

See [`docs/REPRODUCING_RESULTS.md`](docs/REPRODUCING_RESULTS.md) for exact, step-by-step commands covering:
1. Extracting the cached `G_0` (Static Gaussian Generation output) for every scene
2. Training the Dynamic Deformation module
3. Evaluating all three reported configurations (Static, do-nothing, Dynamic)
4. Generating the report's qualitative figures and efficiency benchmark

## Repository Structure

```
Occ4DGS/
├── docs/
│   ├── ARCHITECTURE.md              # Final architecture, module flow, and math
│   ├── RESULTS.md                   # Full results: tables, ablation, qualitative figures
│   ├── REPRODUCING_RESULTS.md       # Step-by-step reproduction guide
│   ├── DATA_AND_CHECKPOINTS.md      # Data/checkpoint download links and layout
│   ├── IMPLEMENTATION_ROADMAP.md    # Development history (Stage A, phases 0-5B)
│   ├── EXPERIMENT_LOG_TEMPLATE.md   # Template used by EXPERIMENT_LOG.md
│   ├── assets/                      # Architecture diagram, result tables, qualitative figures
│   └── deprecated/                  # Superseded design documents, kept for history
├── src/
│   ├── datasets/                    # Dataset adapters and StageBTrainingDataset
│   ├── models/stage_b_temporal/     # Dynamic Deformation module (buffer, encoder,
│   │                                #   deformation heads, transforms)
│   ├── losses/
│   ├── eval/
│   └── deprecated/                  # Superseded, pre-final-design code
├── scripts/
│   ├── baseline_do_nothing.py       # Do-nothing baseline evaluation
│   ├── train_stageb.py              # Dynamic Deformation training loop
│   ├── plot_qualitative.py          # Qualitative BEV comparison figures
│   ├── plot_gaussians.py            # Raw Gaussian primitive visualization
│   └── deprecated/                  # Earlier phases and abandoned exploration
├── configs/
├── experiments/
├── tests/
├── EXPERIMENT_LOG.md                # Full, run-by-run development history
├── GIT_WORKFLOW.md
└── requirements.txt
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@techreport{maung2026occ4dgs,
  title  = {Occ4DGS: Feed-Forward Online Dynamic 3D Gaussian Splatting for
            Occupancy Prediction in Autonomous Driving},
  author = {Maung, La Min},
  institution = {National Chung Cheng University},
  year   = {2026}
}
```

## Acknowledgements

This work reuses [GaussianFormer3D](https://github.com/NVlabs/GaussianFormer3D)'s own reconstruction pipeline and 3D deformable attention mechanism directly for the Static Gaussian Generation stage. This project was supported by the College of Engineering, National Chung Cheng University, Taiwan.