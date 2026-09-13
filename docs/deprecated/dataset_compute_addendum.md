# Addendum: Dataset Plan & Compute Budget
### Based on actual machine findings (RTX 3090 24GB, nuScenes v1.0-mini, Occ3D-nuScenes GT)

*Supersedes dataset assumptions in v2 (which mirrored GaussianFormer3D's SurroundOcc setup).*
*N_g, Stage-1/2 LR ratio, unroll window length, and frozen-vs-joint remain yours to decide —
this addendum only narrows the data source and flags hard compute constraints those decisions
must fit inside.*

---

## 1. Dataset decision: use v1.0-mini, not a trainval subset

`/media/user/Transcend/nuScenes/v1.0-mini` is self-contained (all 6 cameras, LiDAR, radar,
maps, 5.1GB) and **is** the 10 scenes — not a random sample you'd cut from trainval. Use this
directly as your scene set rather than hand-picking 10 scenes out of `data123`'s trainval
blobs. This sidesteps entirely the blob 04/05 corruption and the 17,175-token blacklist you
had to build for QG-Fusion — mini's blob was downloaded separately and is complete.

Two other nuScenes-labeled folders exist on disk (`data/nuscenes` with only `CAM_BACK` +
`LIDAR_TOP`, and `data/nuscenes_cam`) — these look like partial/camera-only extractions from
an earlier project and are **not sufter for this work** (you need all 6 camera views for
GaussianFormer3D's backbone+FPN). Ignore them for this project; don't let path confusion
cost you a debugging session later.

## 2. GT source decision: Occ3D-nuScenes, not SurroundOcc

v2 of the design doc mirrored GaussianFormer3D's SurroundOcc setup (Table I in that paper).
Given what's actually verified and indexed on your machine from the 3DGS static project,
**switch primary GT to Occ3D-nuScenes** (`data/occ3d/gts/`, 850 scene folders, 2.7GB):

- You already have a working index for it (mini_train index offset, blacklist logic,
  coordinate conventions) from the prior project — reuse that loader rather than writing a
  fresh SurroundOcc loader from scratch. This is a real implementation-time saving, not a
  cosmetic preference.
- Occ3D provides a **camera visibility mask**, which GaussianFormer3D's own paper uses in its
  best Occ3D configuration (`GaussianFormer3D*` rows in Table II) — useful for handling
  partially-observed voxels, which matters more here since your temporal chain will
  repeatedly re-observe (or fail to re-observe) the same regions across frames.
- The 850 scene-folder GT set covers the **full trainval scene list**, and nuScenes-mini's 10
  scenes are a curated subset of trainval — so Occ3D GT for all 10 mini scenes should already
  exist inside `data/occ3d/gts/`. Verify this explicitly as a Phase 0 step (match mini's 10
  `scene-XXXX` names against the 850 folders) rather than assuming it — cheap to check, costly
  to discover missing mid-training.
- `data/surroundocc/samples` exists too, so SurroundOcc remains available if you (or the
  professor) later want a second label set for a robustness check — just not the primary path.

**One known gotcha to build around, from your own findings:** Occ3D GT coverage on
`mini_train` starts at **index 39** — indices 0–38 have no GT file. When you build the
per-scene frame index for constructing temporal clips (for the truncated-BPTT unroll window),
explicitly tag each frame with `has_gt: bool` and only slice **contiguous runs of valid-GT
frames** for training clips. Don't assume every frame in a scene has a label — this will
silently corrupt a clip's loss if unchecked (a frame with no GT contributing zero-gradient
`L_occ` for that step is fine; a missing GT file crashing the dataloader mid-epoch is not).

## 3. Coordinate/range alignment check (Phase 0 item)

Your own table gives Occ3D's `pc_range = [-40,-40,-1, 40,40,5.4]`, voxel size `0.4m`, grid
`200×200×16`. GaussianFormer3D's Occ3D config in the paper uses the same range/resolution, so
no conversion needed there. But **do explicitly verify** your voxel-to-Gaussian
initialization (Stage A, §1.3 in v2) uses this same `pc_range` for voxelizing the LiDAR
sweeps — a mismatch here (e.g. accidentally using SurroundOcc's `[-50,50]×[-50,50]×[-5,3]`
range from the paper's default config) would silently misalign Gaussian positions with the
Occ3D voxel grid you're supervising against.

## 4. Compute budget: what the RTX 3090 (24GB, single GPU) forces

The original GaussianFormer3D was trained on A40s (48GB) with `ResNet101-DCN` backbone,
batch size 8, `N_g = 25,600`. A single 24GB 3090 cannot match this directly — plan for the
following adaptations regardless of what you land on for `N_g`/window length:

- **Backbone:** consider `ResNet50 + FPN` instead of `ResNet101-DCN` for the camera branch.
  This also happens to match what you already use in QG-Fusion, so it's both a memory saving
  and a code-reuse win — same backbone class, one fewer thing to debug from scratch.
- **Image resolution:** GaussianFormer3D uses 900×1600. On 24GB with a temporal unroll (Stage
  B reruns the encoder every frame in the window), consider downscaling — this is a genuine
  quality/memory tradeoff worth a quick pilot run rather than guessing.
- **Mixed precision (AMP)** is close to mandatory here, not optional — both Stage A and the
  unrolled Stage B.
- **Gradient checkpointing** — you already used this in QG-Fusion's Phase 2 training loop;
  apply the same technique to both Stage A's 4 refinement blocks and Stage B's per-frame
  encoder calls within the unroll window. This is likely the single highest-leverage change
  for making a multi-frame unroll fit in 24GB at all.
- **Batch size 1 with gradient accumulation** is the realistic starting point for a single
  3090, rather than the paper's batch size 8.

None of the above forces a particular `N_g` or window length — they just set the ceiling
your own choices need to fit under. A reasonable way to pick concretely: run Phase 0/1 (Stage
A alone, no unroll) at your default settings, note peak VRAM, and back-calculate how much
headroom Stage B's per-frame-in-window cost can consume before you commit to a window length.

## 5. Environment compatibility — a Phase 0 risk item, not just a note

Your stack is Python 3.8, CUDA 12.8, spconv 2.3.6 (cu120). GaussianFormer3D's public
codebase (built on an mmdet3d/mmcv-based stack, per the paper's implementation section) may
pin different torch/mmcv/mmdet3d versions than what your QG-Fusion venv currently has.
**Before writing any new code**, do a throwaway test: clone the GaussianFormer3D repo into a
separate venv (or check its `requirements.txt`/`environment.yml` against your existing
package versions) and confirm it at least imports and runs its own demo/inference script on
your machine. This is a 30–60 minute check that prevents discovering a dependency conflict
after you've already written Stage B on top of an incompatible Stage A.

## 6. Updated Phase 0 checklist (concrete, in order)

1. Confirm GaussianFormer3D repo runs in your environment (compatibility check, §5).
2. Build a `scene → [frame_token, has_gt]` index for all 10 mini scenes, using your existing
   Occ3D loader logic from the 3DGS project. Confirm all 10 scene names exist in
   `data/occ3d/gts/`.
3. Verify `pc_range`/voxel resolution match between your Stage A voxelization and the Occ3D
   grid (§3).
4. Run GaussianFormer3D's own Stage A pipeline on **one** mini scene's frame 0, log VRAM peak,
   confirm Gaussian outputs look sane (position scatter plot, no collapse) before scaling to
   all 10 scenes or touching Stage B.
