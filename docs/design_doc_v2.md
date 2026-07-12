# Feedforward Dynamic 3D Gaussian Occupancy Prediction — v2
### Detailed Architecture, Confirmed Design Decisions, Implementation Roadmap

*Prepared for: Liam (La Min Maung) — CCU Autonomous Driving Perception Lab*
*Supersedes v1 — incorporates confirmed answers on recursion and fine-tuning*

---

## 0. Confirmed decisions (locked in from this discussion)

1. **Recursive, buffer-based pipeline.** GaussianFormer3D runs **once**, on frame 0 only, to
   produce the primitive Gaussians `G_0`. From then on, a **reference buffer** always holds
   the *previous deformed output* `G_{t-1}` (never re-anchored back to `G_0`). Each frame's
   deformation writes its output `G_t` both (a) back into the buffer for the next frame, and
   (b) forward into Gaussian-to-voxel splatting for `Ô_t`. This resolves the recursive-vs-anchored
   ambiguity from v1 in favor of **fully recursive**.
2. **Joint fine-tuning of GaussianFormer3D**, staged rather than either fully frozen or
   fully joint from step 0 (justification in §5). This is the chosen novel design point —
   see §5 for the argument and the concrete two-stage training schedule that makes it tractable.

---

## 1. Stage A — GaussianFormer3D primitive Gaussian generation (detailed)

This stage runs **exactly once per sequence**, on frame 0's camera + LiDAR, and its output
`G_0` seeds the entire recursive chain. Because it only runs once but its quality propagates
through every subsequent frame, it is worth implementing faithfully rather than as a black box.

### 1.1 Inputs
- **Multi-view camera images** `I = {I_i}, i=1..N_c` (6 surround views for nuScenes),
  resolution 900×1600 as in the original paper (can downscale for the 10-scene compute budget
  — flag as a tunable if training time is tight).
- **LiDAR point cloud**, aggregated over the most recent `N_f` sweeps (paper uses 10) into a
  combined cloud `P̄ = {P_j}`, each point `(x,y,z,η)` with intensity `η`.

### 1.2 Camera and LiDAR feature encoders
- **Camera branch:** ResNet101-DCN backbone (pretrained from FCOS3D) + FPN neck → multi-scale
  RGB feature maps `F^c`. These encoders are **shared and reused verbatim in Stage B** —
  do not build a separate encoder for the temporal module; this halves your implementation
  surface and is the natural place where joint fine-tuning connects the two stages.
- **LiDAR branch:** voxelize `P̄`, encode via a voxel feature encoder (VoxelNet-style, as
  GaussianFormer3D uses) → multi-scale LiDAR depth maps `F^d`. Also reused in Stage B.

### 1.3 Voxel-to-Gaussian initialization
Voxelize `P̄` at a chosen resolution (paper's ablation favors `0.075m × 0.075m × 0.2m` but
this is a compute/quality knob — start coarser for the 10-scene budget and tighten if time
allows). For each non-empty voxel `v`, compute:
```
m_v = mean{(x,y,z) : points in v}        (position)
σ_v = mean{η : points in v}               (opacity seed)
```
Select `N_g` non-empty voxels (subsample if more voxels than `N_g`, or resample with
replacement if fewer) to initialize Gaussian **means** and **opacities**. `N_g = 25,600` in
the original paper; for a 10-scene budget, **start smaller (e.g. 6,400–12,800)** — the
paper's own ablation (Table IV, GaussianFormer-2 row) shows this only costs a few IoU points
while cutting memory/compute roughly in half, which matters more here than chasing SOTA numbers.

Two Gaussian feature sets are maintained (as in the paper):
- **Physical properties** `G = {G_i ∈ R^d}`, `d = 11 + |C|` — the actual learning target
  (mean, rotation, scale, opacity, semantic logits).
- **High-dimensional query features** `Q = {Q_i ∈ R^m}` — used only as attention queries,
  discarded after Stage A (Stage B builds its own temporal query features, see §2.4).

### 1.4 Sparse convolution self-encoding
Apply a 3D sparse convolution module over the initialized Gaussians (treated as a sparse
point set in 3D) to let nearby Gaussians exchange information before attention — this is a
cheap operation (no dense volume) and directly reusable as a library call
(`spconv`, which you already have working from the QG-Fusion project).

### 1.5 LiDAR-guided 3D deformable attention
This is the mechanism that actually pulls camera semantics into the Gaussians, and its
structure will be **reused with a different attention target in Stage B**, so implement it
as a standalone, swappable module now.

1. Build the unified LiDAR-camera 3D feature space:
   ```
   F^3D = F^d ⊗ F^c        (outer product, multi-scale)
   ```
2. **Stage 1 sampling:** for each Gaussian, shift its mean by learned offsets to get
   `N_R1` 3D reference points `{m + Δm_i}`.
3. Project each reference point into `F^3D` via camera extrinsics/intrinsics to get
   `(u, v, d)` coordinates.
4. **Stage 2 sampling:** at each projected point, generate further learnable offsets
   `(Δu, Δv, Δd)` to get `N_R2` sampling points per reference point.
5. Aggregate via a DFA3D-style deformable attention operator, weighted-summed across all
   camera views, to produce a query update `ΔQ`.

Wrap this two-stage sampling + DFA3D aggregation as a function
`deform_attend(queries, mean_positions, feature_volume, extrinsics, intrinsics) → Δquery`
— Stage B calls the *same function* with a different `feature_volume` (current frame's,
instead of frame 0's), which is the cleanest way to enforce "reuse, don't reimplement."

### 1.6 Iterative refinement
Repeat **4 blocks** of {sparse convolution → LiDAR-guided 3D deformable attention → MLP
property refinement} (paper's default). Each block's MLP decodes the updated query `Q` into
a residual on `G` (position, rotation, scale, opacity, semantics), added to the running estimate.

### 1.7 Output
`G_0 = {μ_i, r_i, s_i, α_i, c_i}_{i=1..N_g}` — handed to the reference buffer as the seed
for Stage B's recursion.

### 1.8 Implementation notes for the 10-scene budget
- Cache camera/LiDAR encoder outputs for frame 0 of each scene to disk once verified stable
  — avoids recomputation across training epochs early on (Phase 0/1), switch to on-the-fly
  once you start jointly fine-tuning (frozen features can't be cached once gradients flow
  through them).
- Log intermediate Gaussians (position scatter plots) after each of the 4 refinement blocks
  during Phase 0 — this is exactly the kind of sanity check that caught the "position
  collapse, Z-axis collapse, scale saturation" bugs you already debugged in QG-Fusion; expect
  the same failure modes here and check for them early rather than after full training runs.

---

## 2. Stage B — Recursive feedforward temporal deformation (detailed)

### 2.1 Reference buffer semantics (confirmed)
A single buffer slot holding the **most recent Gaussians**:
```
buffer ← G_0                          # after Stage A, once
for t = 1 .. T:
    G_t ← deform(buffer, Camera_t, LiDAR_t)     # Stage B, this section
    Ô_t ← splat(G_t)                            # Stage C
    buffer ← G_t                                 # write-back, recursive — never reset to G_0
```
No periodic re-anchoring, no keyframe reset. This is simpler to implement and matches your
instruction directly; the tradeoff (drift over long clips) is handled at the **training**
level via truncated BPTT (§2.6), not by changing the architecture.

### 2.2 Current-frame encoding (reuse Stage A modules)
Run the **same** camera backbone+FPN and **same** LiDAR voxel encoder from §1.2 on
`Camera_t, LiDAR_t` → `F^c_t, F^d_t` → `F^3D_t = F^d_t ⊗ F^c_t`. Reusing these weights
(rather than a fresh encoder) is what makes joint fine-tuning of Stage A meaningfully connect
to Stage B: gradients from `L_occ` at frame `t` flow back through the same encoder that also
built `G_0`, so the encoder is pushed to produce features that are good for *both* initial
generation and repeated temporal querying.

### 2.3 Predicted multi-resolution motion grid `M_t` (the core novelty)
Original 4DGC treats `M_t` as **free parameters**, optimized per-video via gradient descent
at test time (that's what "compression" meant there — you're transmitting fitted parameters).
Here, `M_t` must instead be **predicted by a network** from `F^3D_t` so that the same weights
generalize to new frames and new scenes without any per-scene fitting. Concretely:

```
M_t = HyperNet(pool(F^3D_t))
```
where `HyperNet` is a small 3D-CNN or MLP head that maps the pooled current-frame feature
volume to `L` compact multi-resolution grids `{M_t^l}_{l=1..L}` (`L=3`, matching 4DGC/4D-GS
defaults). This `HyperNet` is the **one genuinely new module** in the whole system — everything
else is reused from GaussianFormer3D (encoders) or 4DGC (grid query mechanism, below).

### 2.4 Per-Gaussian query into the grid
For each Gaussian `i` in the buffer (mean `μ_{t-1}^i`):
```
P_{t-1}^i = {sin(2^l π μ_{t-1}^i), cos(2^l π μ_{t-1}^i)}_{l=1..L}     # positional encoding, 4DGC Eq. 3
z_t^i = concat_l[ trilinear_interp(P_{t-1}^{i,l}, M_t^l) ]            # per-Gaussian temporal feature
```
This is identical in form to 4DGC's motion-grid query (Eq. 3–4), with the sole difference
that `M_t` here is a **predicted tensor**, not a free parameter — the query mechanism itself
doesn't need to change.

### 2.5 Deformation heads `Φ_μ`, `Φ_r`
Two lightweight shared MLPs (shared across all Gaussians and, crucially, **shared across
all frames and all scenes at inference** — this is what "feedforward" buys you over 4DGC):
```
Δμ_t^i = Φ_μ(z_t^i)
Δr_t^i = Φ_r(z_t^i)          # predicted as a small rotation quaternion, e.g. via tanh-bounded axis-angle then quat exponential map
```
No `Φ_s` head (scale), no compensated-Gaussian branch — confirmed dropped.

### 2.6 Update rule and buffer write-back
```
μ_t^i = μ_{t-1}^i + Δμ_t^i
r_t^i  = normalize( Δr_t^i ⊗ r_{t-1}^i )      # quaternion composition, not addition
s_t^i  = s_{t-1}^i,  α_t^i = α_{t-1}^i,  c_t^i = c_{t-1}^i     # time-invariant
G_t = {μ_t^i, r_t^i, s_t^i, α_t^i, c_t^i}
buffer ← G_t
```

**Drift mitigation (training-time, not architecture-time):**
- Truncated backprop through time — unroll 3–5 frames per training step, detach the buffer's
  gradient history beyond that window (`.detach()` at the truncation boundary) rather than
  back-propagating through the full clip.
- Log per-frame mIoU across a full unrolled validation clip (not just single-step) during
  development — this is your direct signal for whether drift is a real problem in practice
  before you spend time engineering around it.

---

## 3. Stage C — Gaussian-to-voxel splatting → occupancy (unchanged from v1)

Unchanged from GaussianFormer3D's own splatting module: evaluate the Gaussian mixture at
each voxel center within a local neighborhood of contributing Gaussians, sum weighted
contributions, produce `Ô_t ∈ C^{X×Y×Z}`. No changes needed here — it operates identically
on `G_t` from Stage B as it would on any static Gaussian set from Stage A.

---

## 4. Loss function (unchanged from v1, restated briefly)

```
L = λ_occ · L_occ(Ô_t, O_t^gt)  +  λ_tv · L_tv(Δμ_t, Δμ_{t-1}, Δr_t, Δr_{t-1})  +  λ_lidar · L_lidar(G_t, P_t)
```
- `L_occ`: CE + Lovász-softmax vs SurroundOcc GT, averaged over all unrolled frames.
- `L_tv`: penalizes **change in predicted motion** frame-to-frame (acceleration, not motion
  itself) — see v1 §3.2 for the exact form; unaffected by the recursion decision above.
- `L_lidar`: nearest-Gaussian or depth-consistency term between `μ_t` and `P_t` — unaffected.

---

## 5. Frozen vs. fine-tuned GaussianFormer3D — decision and justification

**Chosen design: staged joint fine-tuning**, not a binary frozen/unfrozen choice. Reasoning:

Because the recursion is fully confirmed-recursive (§2.1), `G_0`'s quality is not just a
frame-0 concern — it is the seed that every later frame's occupancy accuracy depends on
*through the entire chain*. A `G_0` optimized only for single-frame occupancy (the frozen-checkpoint
option) has no incentive to produce Gaussians that deform *well* — e.g. Gaussians whose
positions/rotations are well-conditioned for the temporal query in §2.4. Jointly fine-tuning
lets the gradient from `L_occ` at *every* frame `t` flow back through the deformation chain
into Stage A's encoders and refinement blocks, teaching Stage A to produce Gaussians that are
good **temporal seeds**, not just good single-frame reconstructions. This is also a genuinely
reportable finding for the paper ("does joint fine-tuning of the static generator improve
temporal propagation quality, vs. a frozen generator?") — worth keeping frozen-vs-joint as an
**ablation**, but joint fine-tuning as the **main reported configuration**.

**Two-stage training schedule** (mitigates the instability risk of jointly training an
untrained new module — `HyperNet`, `Φ_μ`, `Φ_r` — alongside an already-pretrained one):

- **Stage 1 (warm-up, frozen Stage A):** load GaussianFormer3D from a pretrained checkpoint,
  freeze it entirely. Train only `HyperNet`, `Φ_μ`, `Φ_r` using the full loss over short
  unrolled windows. This gives the new temporal module a stable, non-moving target to learn
  against — directly analogous to 4D-GS's 3D-Gaussian warm-up (§4.3 in that paper) and
  TED-4DGS's static-anchors-first progressive training, both of which you've already read.
- **Stage 2 (joint fine-tune):** unfreeze GaussianFormer3D's weights, continue training
  end-to-end with a **smaller learning rate for Stage A** (e.g. 5–10× lower than Stage B's
  LR) so the already-converged generator adapts gradually rather than being disrupted by
  large early gradients from the still-adapting temporal module.
- Report both the Stage-1-only (frozen) and Stage-2 (joint) checkpoints as an ablation row —
  this directly answers "was joint fine-tuning worth it" with a number, not just an argument.

---

## 6. Updated implementation roadmap

**Phase 0 — Stage A standalone.** Get GaussianFormer3D (§1) fully working and validated
(reproduce paper's rough IoU/mIoU ballpark on a couple of scenes) before touching Stage B at
all. Cache `G_0` per scene.

**Phase 1 — Stage C wiring smoke test.** Feed a cached `G_0` directly into splatting (Stage C)
with zero deformation, confirm occupancy loss and gradients behave — this reuses Phase 0's
output and needs no new modules yet.

**Phase 2 — Stage B skeleton.** Implement `HyperNet → grid query → Φ_μ, Φ_r → update rule`,
initially with dummy/zero encoders, to validate shapes and the recursion loop (buffer
read/write) on a 2-frame toy sequence.

**Phase 3 — Real encoders + Stage 1 training.** Wire in the real (frozen) Stage A encoders,
train `HyperNet, Φ_μ, Φ_r` per §5 Stage 1, on short unrolled windows (2–3 frames), 1–2 scenes
first, then scale to ~10.

**Phase 4 — Loss completion.** Add `L_tv` and `L_lidar`, ablate each to zero to confirm
expected failure modes (jitter without `L_tv`; floating/inconsistent geometry without `L_lidar`).

**Phase 5 — Stage 2 joint fine-tuning.** Unfreeze Stage A per §5's schedule, extend unroll
window (3–5 frames), scale to full ~10-scene training set.

**Phase 6 — Evaluation & ablations.** Per-frame IoU/mIoU on held-out frames/scenes; temporal
flicker metric; frozen-vs-joint ablation (§5); `L_tv`/`L_lidar` ablations (§4); `N_g` size
ablation (§1.3) if time allows.

**Phase 7 — Paper writing** (outline unchanged from v1 §6 — Introduction, Related Work,
Method [Stages A/B/C as above], Experiments, Ablations, Limitations, Conclusion). Update the
Method section particularly to make the "reused encoders + one new HyperNet module" framing
explicit — it's a clean, defensible novelty story: *the only new parameters are the motion
hypernetwork and two small deformation heads; everything else is inherited machinery, made
temporal by construction.*

---

## 7. What to double check with the professor next

1. Confirm `N_g` (number of Gaussians) target for the 10-scene compute budget — this is a
   pure compute/quality tradeoff, not a novelty question, so it's a quick decision.
2. Confirm the Stage 1 → Stage 2 learning-rate ratio and unroll-window lengths — these are
   the two hyperparameters most likely to need a couple of pilot runs before the full sweep.
3. Confirm whether the frozen-vs-joint ablation (§5) should be a main table or a smaller
   appendix ablation, depending on how much space the target venue gives you.
