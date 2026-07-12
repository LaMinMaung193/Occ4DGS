# Feedforward Dynamic 3D Gaussian Occupancy Prediction
### Design Document — Post-Meeting Architecture with Prof. Rachael Chiang

*Prepared for: Liam (La Min Maung) — CCU Autonomous Driving Perception Lab*
*Scope target: ~10 nuScenes scenes, no compression, feedforward temporal model*

---

## 1. What changed after the meeting

Your earlier designs (Weeks 4–8, and the compression-heavy sketch) treated 4DGC as a
**per-scene, test-time-optimized compression method**: a motion grid `M_t` and compensated
Gaussians were *fitted* to one video, then entropy-coded for storage/transmission.

The new direction drops the compression framing entirely and turns 4DGC's idea into a
**generalizable, feedforward temporal module**, closer in spirit to GaussianFormer3D itself:

| | Old sketch (4DGC-as-compression) | New direction (this paper) |
|---|---|---|
| Motion grid `M_t` | Learned per-scene parameters, transmitted in bitstream | Predicted on-the-fly by a network from current camera+LiDAR |
| Generalization | None — refit per video | Trained once, run on any scene/frame |
| Compensated Gaussians `ΔG_t` | Present (handles disocclusion) | **Dropped** |
| Compression / entropy coding | Core contribution | **Dropped** (explicit future work) |
| Reference for warping | Reconstructed previous frame from decoder | Previous frame's Gaussians (or `G_0` for frame 1) |
| Loss | Rate-distortion (color + bitrate) | Occupancy CE+Lovász, temporal TV, LiDAR consistency |
| Output modality | Rendered RGB (novel-view synthesis) | 3D semantic occupancy grid |

This is the right simplification for a first paper: it isolates **one clean claim** —
*"a feedforward, per-frame-conditioned Gaussian deformation network can propagate a
LiDAR-camera Gaussian scene representation through time for occupancy prediction, without
per-scene optimization or explicit compensation Gaussians."* Compression becomes a stated
limitation / future-work paragraph, which is honest and keeps scope small enough for ~10 scenes.

---

## 2. Full architecture

### 2.1 Stage A — Reference Gaussian initialization (run once, frozen or fine-tuned)

- Use **GaussianFormer3D** exactly as published: voxel-to-Gaussian initialization from
  aggregated LiDAR sweeps + LiDAR-guided 3D deformable attention over camera features.
- Run once per sequence on frame 0 (or once per short clip) to produce `G_0 = {μ_i, r_i, s_i, α_i, c_i}` for `i = 1..N_g`.
- Decide early: is GaussianFormer3D **frozen** (pretrained checkpoint, no gradient) or
  **fine-tuned jointly** with the temporal module? Recommendation: **freeze it for v1**.
  This isolates the temporal module as the sole new contribution, keeps training cheap on
  ~10 scenes, and gives you a clean ablation ("frozen G_0 vs jointly fine-tuned G_0") for v2/rebuttal.

### 2.2 Stage B — Feedforward temporal deformation network (the new contribution)

For every subsequent frame `t = 1 .. T`:

**Inputs:** current-frame camera images `I_t`, current-frame LiDAR point cloud `P_t`, and the
Gaussians from the previous step `G_{t-1}` (recursive; `G_0` seeds `t=1`).

**Step 1 — Encode current sensors.**
Reuse the same camera backbone + FPN and the same LiDAR voxel encoder GaussianFormer3D already
uses (don't invent a new encoder — this keeps parameter count and training cost down and gives
you a fair ablation against "re-running GaussianFormer3D every frame").

**Step 2 — Predict motion grid `M_t`.**
Unlike original 4DGC (which *learns* `M_t` as free parameters via per-scene test-time
optimization), here `M_t` is the **output of a network** conditioned on the fused
camera+LiDAR features of frame `t`. Two reasonable parameterizations — treat this as your
first real design decision / ablation axis:

- **(a) Multi-resolution motion grid + trilinear interpolation** (closer to 4DGC/4D-GS):
  a small hyper-network regresses a compact `L`-level grid `{M_t^l}` from the pooled
  frame features; each Gaussian's mean `μ_{t-1}` queries the grid via positional encoding +
  interpolation, as in 4DGC Eq. 3–4.
- **(b) Direct per-Gaussian deformable attention** (closer to GaussianFormer3D's own
  attention, cheaper to justify architecturally): each Gaussian query attends to the
  current frame's camera/LiDAR feature volume directly (reusing GaussianFormer3D's
  deformable-attention block) and regresses `(Δμ, Δr)` without an explicit grid.

  Recommendation: start with **(b)** — it reuses machinery you already have working code
  intuition for from GaussianFormer3D, avoids grid-resolution hyperparameter search, and is
  easier to describe as "the same attention block, now conditioned on the *new* frame instead
  of re-initializing." Keep (a) as an ablation/alternative if (b) underperforms.

**Step 3 — Two lightweight MLP heads, `Φ_μ` and `Φ_r` only.**
No `Φ_s` (scale) head, no compensated-Gaussian branch — per the professor's simplification.
Rotation and scale of newly *disoccluded* regions are handled implicitly by warping existing
Gaussians rather than spawning new ones; flag this as a known limitation for fast, large
motion / newly-revealed regions (this is exactly the gap 4DGC's compensated Gaussians were
built to patch — worth one sentence in Limitations).

```
Δμ_t, Δr_t = Φ_μ(z_t), Φ_r(z_t)      where z_t = fused per-Gaussian temporal feature
μ_t = μ_{t-1} + Δμ_t
r_t = Δr_t ⊗ r_{t-1}                  (quaternion composition, not addition)
s_t = s_{t-1}, α_t = α_{t-1}, c_t = c_{t-1}   (time-invariant, as in GaussianFormer3D/4DGC)
```

### 2.3 Stage C — Gaussian-to-voxel splatting → occupancy

Identical to GaussianFormer3D's Gaussian-to-voxel splatting module (Eq. 1–3 in that paper):
evaluate the Gaussian mixture at each voxel center within a local neighborhood, sum
contributions, produce `Ô_t ∈ C^{X×Y×Z}`.

### 2.4 Training vs inference — important simplification to lock down

You stated explicitly: **camera + LiDAR of the current frame is the input at both training
and inference.** This is good — it means there is **no teacher forcing gap** and **no
train/test mismatch**, unlike autoregressive video models that sometimes condition on GT
frames during training only. Two consequences to design around:

1. **Error accumulation is a real risk** if `G_{t-1}` is always the *predicted* previous
   Gaussians (fully recursive/autoregressive chain). Over a 10–20 frame clip this could
   drift. Mitigate with:
   - Truncated backprop through time (BPTT) — e.g. unroll 3–5 frames per training step,
     not the full sequence.
   - Optionally reset to a fresh `G_0` (re-run GaussianFormer3D) every `K` frames — a
     "keyframe" every K frames, but **without** any compression/entropy angle, purely to
     bound drift. This is a legitimate design knob to discuss with the professor: fully
     recursive vs periodic re-anchoring to `G_0`.
2. **Ambiguity in your notes** ("based on reference gaussian past frame **or** we got from
   GaussianFormer3D initially") suggests this exact question isn't fully settled yet.
   Recommend making it an explicit ablation: **(i) recursive** (`G_{t-1} → G_t`, drift risk,
   captures compounding motion) vs **(ii) anchored** (`G_0 → G_t` always, deformation is
   absolute displacement from frame 0, no drift but breaks down over long/large motion).
   This ablation alone can be a solid table in the paper and directly resolves the ambiguity.

---

## 3. Loss function (exactly three terms, as instructed)

```
L = λ_occ · L_occ  +  λ_tv · L_tv  +  λ_lidar · L_lidar
```

### 3.1 `L_occ` — occupancy loss
Cross-entropy + Lovász-softmax against SurroundOcc ground truth, applied per frame:
```
L_occ = L_ce(Ô_t, O_t^gt) + L_lovasz(Ô_t, O_t^gt)
```
(Identical formulation to GaussianFormer3D §III-A / SurroundOcc.) Average over all frames
in the clip, not just the last one — otherwise gradient signal for the temporal module is weak.

### 3.2 `L_tv` — temporal smoothness (adapted from TED-4DGS)
TED-4DGS regularizes the *deformation bank* `Z` for scale-consistency (`L_vol`) and temporal
smoothness (`L_tv`) of learned deformation vectors. Here, since `M_t` is predicted (not a
free parameter bank), `L_tv` becomes a smoothness penalty on **consecutive predicted
motions**, discouraging jitter/flicker while still allowing genuine motion:
```
L_tv = (1/N_g) Σ_i || Δμ_t^i − Δμ_{t-1}^i ||_1  +  || Δr_t^i − Δr_{t-1}^i ||_1
```
i.e. penalize the *second derivative* of position/rotation (acceleration), not the motion
itself — this is the key difference from naively penalizing `Δμ_t` toward zero, which would
suppress real motion. State this distinction explicitly in the paper; it's a legitimate,
citable adaptation of TED-4DGS's TV loss to a feedforward (non-bank-based) setting.

### 3.3 `L_lidar` — LiDAR geometric consistency
Enforces that the warped Gaussian means `μ_t` remain consistent with the observed LiDAR
point cloud `P_t` at frame `t` — a per-frame counterpart to GaussianFormer3D's use of LiDAR
only at *initialization*. A simple, defensible instantiation:
```
L_lidar = (1/|P_t|) Σ_{p∈P_t} min_i || p − μ_t^i ||^2      (nearest-Gaussian Chamfer-style term)
```
or, cheaper, a depth-consistency term comparing rendered/splatted depth from `{μ_t}` against
LiDAR range images (reusing the depth-map machinery GaussianFormer3D already builds for its
LiDAR-guided attention). Pick whichever reuses existing code from Stage A most directly.

### 3.4 Weighting
Start with `λ_occ = 1.0`, `λ_tv` and `λ_lidar` swept over `{0.01, 0.1, 1.0}` on a small
validation split before committing — this is a 30-minute experiment, do it before the full run.

---

## 4. Implementation roadmap (phased, incremental — matches your usual working style)

**Phase 0 — Environment & reference Gaussians**
- Get GaussianFormer3D running end-to-end on your ~10 chosen nuScenes scenes; cache `G_0`
  (and intermediate camera/LiDAR features if reusable) to disk per scene.
- Decide and lock: frozen vs fine-tuned GaussianFormer3D (§2.1).

**Phase 1 — Skeleton smoke test**
- Dummy temporal module (e.g. `Δμ=0, Δr=identity`) → warp → splat → occupancy, on 1 scene,
  1 frame pair, verify shapes and gradient flow through Stage C. This confirms the
  Gaussian-to-voxel splatting and loss wiring before any real temporal modeling exists.

**Phase 2 — Real temporal module (variant (b), deformable-attention)**
- Implement the per-Gaussian query → current-frame feature attention → `Φ_μ, Φ_r` heads.
- Overfit sanity test on a single 2-frame pair (can the model deform `G_0` to match frame 1's
  occupancy GT at all?) before scaling to full clips.

**Phase 3 — Losses**
- Add `L_tv` (needs `Δ_{t-1}` cached) and `L_lidar`. Verify each loss independently by
  ablating it to zero and checking the expected failure mode (e.g. no `L_tv` → visibly jittery
  point-cloud trajectories, as in TED-4DGS Fig. 6).

**Phase 4 — Multi-frame training over ~10 scenes**
- Truncated BPTT window (start with 3 frames), batch across scenes.
- Decide recursive vs anchored-to-`G_0` (§2.4) — run both as your first real ablation.

**Phase 5 — Evaluation**
- Per-frame IoU/mIoU vs SurroundOcc GT, averaged over held-out frames of your 10 scenes
  (or held-out scenes if you have enough — 10 is small, consider frame-level held-out split
  within scenes, or leave-one-scene-out cross-validation given the small N).
- Temporal consistency metric: frame-to-frame flicker (e.g. voxel-label change rate at
  static regions) — an easy, cheap metric that supports the `L_tv` story.
- Latency/FPS — since there's no compression, this is now a *speed* story: compare against
  running full GaussianFormer3D independently every frame.

**Phase 6 — Ablations**
- Recursive vs anchored reference (§2.4).
- With/without `L_tv`, with/without `L_lidar`.
- Variant (a) grid-based vs (b) attention-based motion prediction.
- Frozen vs fine-tuned `G_0`.

**Phase 7 — Paper writing** (see §6).

---

## 5. Experimental design summary

| Axis | Choice |
|---|---|
| Dataset | nuScenes, ~10 scenes, SurroundOcc annotations (18-class, matches GaussianFormer3D Table I) |
| Baselines | (1) GaussianFormer3D run independently per frame (no temporal model — the "no-memory" baseline), (2) this method, (3) optionally a naive linear-motion baseline (constant velocity on Gaussian means) as a sanity floor |
| Main metrics | IoU, mIoU (per-class + overall), following GaussianFormer3D's protocol |
| Secondary metrics | temporal flicker rate, inference FPS/latency, memory footprint (Gaussians only, no bitstream) |
| Ablations | §4 Phase 6 table above |

Given the honest ~10-scene scope, be upfront in the paper that this is a **feasibility /
architecture study**, not a claim of SOTA on full nuScenes val — that framing fits smaller
venues (workshop, ICRA/IROS short paper) better than a top-tier main-track claim, and avoids
reviewers asking "why only 10 scenes" as a fatal objection rather than a scope note.

---

## 6. Paper writing plan

**Working title options:**
- "Feedforward Temporal Deformation for Dynamic Gaussian Semantic Occupancy Prediction"
- "From Static to Dynamic Gaussians: A Feedforward LiDAR-Camera Fusion Approach to 4D Occupancy"

**Section outline:**
1. Introduction — motivate: GaussianFormer3D is static/per-frame; 4DGC-style temporal
   modeling exists but is scene-overfit and compression-oriented; gap = a *generalizable*
   feedforward temporal Gaussian model for occupancy.
2. Related work — GaussianFormer3D, GaussianFormer/-2, 4DGC, 4D-GS, TED-4DGS, SDGOCC/SparseLIF.
   Position clearly: "unlike 4DGC/TED-4DGS which optimize per-scene for compression, we train
   one feedforward network across scenes for prediction."
3. Method — Stages A/B/C as above, loss (§3), explicitly state what's dropped (compensated
   Gaussians, entropy coding, keyframe loss) and why (scope decision, future work).
4. Experiments — §5.
5. Ablations — §4 Phase 6.
6. Limitations — no disocclusion handling (no compensated Gaussians), small scene count,
   frozen G_0, no compression (explicit future direction).
7. Conclusion.

**Timeline suggestion:** Phases 0–3 (~3–4 weeks), Phase 4–5 (~2–3 weeks), Phase 6 (~1–2
weeks), writing in parallel from Phase 4 onward rather than after everything is done.

**Venue:** given 10-scene scope and no compression claim, workshop papers (ICRA/IROS
workshops, CVPR workshops on occupancy/autonomous driving) or a short IROS/ICRA paper are
more realistic first targets than a CVPR/ICCV main track submission — worth confirming with
Prof. Chiang once Phase 4 results exist, since the target venue affects how much ablation
depth is expected.

---

## 7. Open questions to bring back to Prof. Chiang

1. **Recursive vs anchored** reference Gaussians (§2.4) — your notes leave this ambiguous;
   worth resolving explicitly since it changes the training loop structure.
2. **Frozen vs fine-tuned** GaussianFormer3D for `G_0` (§2.1).
3. Variant (a) grid-based vs (b) attention-based motion prediction (§2.2) — which does she
   want as the "headline" design, with the other as ablation?
4. Confirm the exact SurroundOcc/Occ3D split to use for the 10 scenes (val-set subset vs a
   custom split), and whether cross-validation (leave-one-scene-out) is expected given N=10.
5. Confirm intended venue early — it changes how much of Phase 6 is necessary before submission.
