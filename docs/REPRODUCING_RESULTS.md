# Reproducing the Results

This guide walks through every step from raw data to every result reported in the final report, in order. See [`DATA_AND_CHECKPOINTS.md`](DATA_AND_CHECKPOINTS.md) first to get all required data/checkpoints in place.

All commands assume both `Occ4DGS` and `GaussianFormer3D` are cloned as sibling directories, and the `gf3d` conda environment is active.

## Step 0: Prerequisites

- nuScenes v1.0-trainval + SurroundOcc annotations in place (`GaussianFormer3D/data/`)
- `GaussianFormer3D`'s own released Stage A checkpoint (`surroundocc_release.pth`) in place
- Depth ground-truth generated (or downloaded)

```bash
cd Occ4DGS
python scripts/generate_depth_gt_full.py
```

## Step 1: Extract Cached G_0 (Static Gaussian Generation Output)

Runs Stage A once per scene, caching its output so no later step needs to re-run it.

```bash
cd GaussianFormer3D
python occ4dgs_scripts/extract_g0_cache.py \
    --checkpoint out/nuscenes_surroundocc_gs25600_full/surroundocc_release.pth
```

Produces `g0_cache_pretrained_release/` (~2.3 GB, 850 scenes).

## Step 2: Build the Stage B Manifest

Finds each scene's frame-0-to-next-genuinely-moving-keyframe pair (≥0.5m ego translation), keeping train and val genuinely separate.

```bash
cd Occ4DGS
python scripts/build_stageb_manifest.py
```

Produces `nuscenes_infos_gf3d_stageb_pairs_{train,val}.pkl` and `stageb_manifest_{train,val}.json` — 651 training pairs, 140 held-out validation pairs.

## Step 3: Train the Dynamic Deformation Module (L=2)

```bash
cd Occ4DGS
python scripts/train_stageb.py
```

Trains for 40 epochs (~11.1 hours on a single RTX 3090). Checkpoints saved every 3 epochs to `checkpoints_L2_pretrained_release/`. The official, reported result is `epoch_24.pth` (val mIoU 18.44).

**For the ablation study (L=4):** edit `NUM_BLOCKS = 4` and `OUT_DIR` in `scripts/train_stageb.py`, then re-run. Trains for ~13.4 hours; the reported result is `epoch_33.pth` (val mIoU 18.12).

## Step 4: Evaluate All Three Configurations

**Table 1 — Static 3DGS (fresh, per-frame reconstruction):**
```bash
cd GaussianFormer3D
python occ4dgs_scripts/eval_static_stageA_per_frame.py
```
Expected: `mIoU: 25.13`

**Table 2 — Do-nothing baseline:**
```bash
cd Occ4DGS
python scripts/baseline_do_nothing.py
```
Expected: `mIoU: 16.75`

**Table 3 — Dynamic Deformation (ours):**
```bash
cd GaussianFormer3D
python occ4dgs_scripts/eval_stageb_checkpoint.py \
    --checkpoint <path-to-checkpoints_L2_pretrained_release/epoch_24.pth> \
    --num_blocks 2 \
    --out eval_results_L2.json
```
Expected: `mIoU: 18.44`

**Table 5 — Ablation (L=4):**
```bash
cd GaussianFormer3D
python occ4dgs_scripts/eval_stageb_checkpoint.py \
    --checkpoint <path-to-checkpoints_L4_pretrained_release/epoch_33.pth> \
    --num_blocks 4 \
    --out eval_results_L4.json
```
Expected: `mIoU: 18.12`

## Step 5: Generate Qualitative Figures (Figure 4)

```bash
cd GaussianFormer3D
python occ4dgs_scripts/render_qualitative.py
```

Then, from `Occ4DGS`:
```bash
cd Occ4DGS
python scripts/plot_qualitative.py
```

## Step 6: Efficiency Benchmark (Table 4)

```bash
cd GaussianFormer3D
python occ4dgs_scripts/benchmark_efficiency.py
```

Measures real inference latency, peak VRAM, and parameter count for Static 3DGS and Dynamic Deformation (L=2 and L=4), averaged over 20 held-out scenes.

## Note on the Released Static Checkpoint

This project adopts GaussianFormer3D's own publicly released checkpoint as the Static Generation module, rather than training Stage A independently from scratch. See the report's own Training Setting section, and [`docs/IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md) (Phase 5B) for the reasoning behind this choice.