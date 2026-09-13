# Occ4DGS: Architecture and Design

This document describes the final, adopted design of the Dynamic Deformation
module (Stage B), which propagates a scene's Gaussian representation forward
in time via a lightweight, feedforward network that reuses GaussianFormer3D's
own real architecture directly. It is a companion to Figure 1 (the
architecture diagram) and the paper's own Method section (Section 3).

## 1. Overview

At $t=0$, the scene's Gaussian representation is produced by GaussianFormer3D's
own, unmodified reconstruction pipeline, run once, producing $G_0$. Every
subsequent frame ($t>0$) runs a feedforward deformation step that reuses
GaussianFormer3D's own real modules directly — not reimplementations — applied
to $G_{t-1}$ (the reference buffer's current Gaussian state) instead of a
freshly-initialized one.

**Final-commit strategy.** Every block's deformation heads output is applied
directly to that block's own running anchor, and the anchor itself — not a
separately-tracked delta — carries forward into the next block's attention.
After the last block ($l=L$), $G_t = \text{anchor}^{(L)}$ directly. This
matches GaussianFormer3D's own real per-block behavior exactly: its own refine
step updates the anchor every block and feeds the updated anchor into the next
block's key-point generator, with no separate "transient vs. final"
distinction anywhere in its real code. This design carries no open validation
question, since there is no mismatch between what grounds the query's search
and what gets committed — they are the same anchor, throughout.

## 2. Module-by-Module Flow ($t>0$)

**Step 1 — Backbones (reused, unchanged).** Input: frame $t$'s 6 camera images
+ LiDAR point cloud. Output: $F^c_t$ (multi-scale camera features), $F^d_t$
(multi-scale depth-conditioned features), via the same `CurrentFrameEncoder`.

**Step 2 — Outer Product / KV construction (reused, unchanged).** $F^{3D} =
F^d \otimes F^c$ describes what `DeformableFeatureAggregation3D`'s CUDA kernel
achieves functionally, not a separate, materialized step. $F^c_t / F^d_t$ are
passed directly into Step 4 as `feature_maps`/`dpt_feature_maps`.

**Step 3 — Query and anchor construction.** Two tensors feed every block, each
with a different role:

- **Anchor** — the Gaussian's explicit properties (position, scale, rotation,
  opacity, class; 28 values, all interpretable: 3 (position) + 4 (rotation) +
  3 (scale) + 1 (opacity) + 17 (semantic classes) = 28). Comes directly from
  the reference buffer, $G_{t-1}$, at block 1; evolves as the real, cascading
  Gaussian state block-to-block — the actual position/rotation the model is
  converging toward $G_t$ itself, not a transient value discarded at the end.
  Tells the model where the Gaussian currently is.
- **Q (instance feature)** — an abstract, high-dimensional working state (not
  interpretable). Starts as a fixed, learned, per-Gaussian-slot value,
  independent of the anchor's specific values, matching GaussianFormer3D's own
  design: this table is random at the start of training, shaped by gradient
  descent over training, and frozen as trained values once training finishes;
  at inference it is the same fixed starting point for every scene, with each
  scene's own information entering only afterward via attention. Decides how
  much to trust, and where exactly, to sample.
- **Combining them** — before each block's attention, the anchor is embedded
  into the same high-dimensional space as Q (`anchor_embed = AnchorEncoder(anchor)`),
  then added directly to Q. The anchor encoder's real internal structure does
  not project the flat 28-dim vector in one pass — it splits the anchor into
  five separate property groups (position, scale, rotation, opacity, class),
  encodes each independently into the embedding dimension via its own small
  projection stack, sums the five results, then applies one final output
  projection.

**Step 4 — 3D Deformable Attention (reused verbatim,
`DeformableFeatureAggregation3D`).** Input: instance feature (Q), anchor,
anchor embedding, feature maps ($F^c_t$), depth feature maps ($F^d_t$), and
metadata (including the projection matrix and image dimensions). Internally:

1. Reference key points are generated from the anchor's own scale/rotation/
   position (a fixed geometric template), concatenated with an additional,
   learnable-scale template derived from the instance feature — so key points
   are anchor-dominant but not exclusively anchor-derived.
2. These key points are projected into normalized $(u, v, \text{depth})$
   space; depth is preserved as a real coordinate throughout, rather than
   being discarded after projection.
3. Sampling weights are computed from the combined instance-feature-plus-
   anchor-embedding signal, softmaxed per camera over levels and points.
4. Learnable sampling offsets, derived from the same combined signal, are
   added directly onto the already-projected points — a two-stage sampling
   refinement (anchor/Q-derived geometric reference points, then Q-and-anchor-
   derived learned offsets around each of those points).
5. The low-level attention kernel samples the fused feature volume at the
   final points, weights, and sums — the only part of the pipeline that is a
   literal "read the feature map, weight it, sum" operation; everything
   determining where to sample and how much to trust each sample happens
   beforehand, using the instance feature, anchor, and anchor embedding as
   real inputs.
6. The output is concatenated with the block's own input query (not added),
   giving a doubled-dimension result — this is why the feedforward network in
   Step 5 reduces back down to the original dimension, rather than the
   attention module's own output already matching it.

**Step 5 — FFN + LayerNorm (reused, unchanged).** The concatenated output from
Step 4 passes through a feedforward network (reducing back to the original
embedding dimension) followed by one layer normalization. Attention alone
produces only a weighted sum of sampled values — a linear combination that can
interpolate between what it's given but never create a genuinely new pattern;
the feedforward network adds non-linear processing on top, independently per
Gaussian. Layer normalization keeps the query's values from drifting to
extreme magnitudes after the residual-style concatenation that precedes it.

**Step 6 — Deformation Heads (project-specific).** Input: the block's query
output, together with the anchor embedding — matching the real input
signature GaussianFormer3D's own refinement module uses (not the query
alone). Output: a residual position offset and rotation update per Gaussian,
via two lightweight, tanh-bounded heads.

**Step 7 — Anchor update, every block.** The anchor's position and rotation
are updated directly by this block's residual offsets; scale, opacity, and
semantics remain frozen throughout (time-invariant under this design's update
rule). The anchor embedding is recomputed fresh from the updated anchor,
ready for the next block's attention. This update is real and cumulative —
not a transient value used only to seed search and then discarded. After the
final block, the anchor itself, in its current state, becomes $G_t$ directly.

## 3. Math

### 3.1 Setup

$$G_{t-1} = \{\mu_i, r_i, s_i, \alpha_i, c_i\}, \quad i = 1..N_g \quad \text{(reference buffer)}$$
$$\text{anchor}^{(0)}_i = \text{concat}(\mu_i, s_i, r_i, \alpha_i, c_i) \quad \text{(28-dim, per Gaussian)}$$

### 3.2 Query and Anchor Initialization

$$Q^{(0)} = \text{InstanceFeatureEmbedding} \quad \text{(fixed, learned, per-slot; independent of anchor)}$$
$$\text{anchorEmbed}^{(0)} = \text{AnchorEncoder}(\text{anchor}^{(0)})$$

### 3.3 Frame Transform Before Projection

The anchor's position ($\mu_i$) and rotation ($r_i$) are stored in
$G_{t-1}$'s own frame. The deformable attention module's projection step
requires positions already in frame $t$'s local coordinate frame, matching
frame $t$'s own projection matrix. The key-point generator also uses the
anchor's rotation to orient its sampling template (an oriented ellipsoid, not
a sphere) — so the same frame-consistency requirement applies to rotation,
not just position: if the ego vehicle turned between $t-1$ and $t$,
"up/forward/right" in frame $t-1$'s terms is not the same as in frame $t$'s
terms. Leaving rotation unconverted while converting only position would
build an internally inconsistent template.

Before the anchor is passed into the deformable attention module, a
transient (not persisted) copy of both its position and rotation is
transformed:

$$T = \text{computeRelativeTransform}(\text{pose}_{prev}, \text{pose}_{curr})$$
$$\mu_i^{(\text{proj})} = T \cdot [\mu_i, 1]^T$$
$$q_{\text{rel}} = \text{rotmatToQuat}(T[:3,:3])$$
$$r_i^{(\text{proj})} = \text{quatMultiply}(q_{\text{rel}}, r_i)$$

Only the transient, projection-input copy is affected — $G_{t-1}$'s actual
stored $\mu_i$, $r_i$ are never reassigned to a new frame; this transform is
applied purely for projection purposes, not as a persistent ego-compensation
of the buffer's own stored state.

### 3.4 Per-Block Iteration, $l = 1..L$

$$Q_{\text{cat}}^{(l)} = \text{DeformableFeatureAggregation3D}(Q^{(l-1)}, \text{anchor}^{(l-1)}, \text{anchorEmbed}^{(l-1)}, F^c_t, F^d_t, \text{metas})$$

$$Q^{(l)} = \text{LayerNorm}(\text{FFN}(Q_{\text{cat}}^{(l)}))$$

$$\Delta\mu_i^{(l)} = \Phi_\mu(Q_i^{(l)}, \text{anchorEmbed}_i^{(l-1)}), \qquad \Delta r_i^{(l)} = \Phi_r(Q_i^{(l)}, \text{anchorEmbed}_i^{(l-1)})$$

$$\text{anchor}^{(l)}_\mu = \text{anchor}^{(l-1)}_\mu + \Delta\mu_i^{(l)}$$
$$\text{anchor}^{(l)}_r = \text{normalize}(\text{quatMultiply}(\text{anchor}^{(l-1)}_r, \Delta r_i^{(l)})) \quad \text{(current-first)}$$
$$\text{anchor}^{(l)}_s, \text{anchor}^{(l)}_\alpha, \text{anchor}^{(l)}_c = \text{anchor}^{(l-1)}_s, \text{anchor}^{(l-1)}_\alpha, \text{anchor}^{(l-1)}_c \quad \text{(frozen)}$$
$$\text{anchorEmbed}^{(l)} = \text{AnchorEncoder}(\text{anchor}^{(l)})$$

### 3.5 Final Update

$$G_t = \text{anchor}^{(L)}$$

That is: $\mu_{t,i} = \mu_i^{(L)}$, $r_{t,i} = r_i^{(L)}$,
$s_{t,i} = s_{t-1,i}$, $\alpha_{t,i} = \alpha_{t-1,i}$, $c_{t,i} = c_{t-1,i}$ (where $\mu_i^{(L)}$, $r_i^{(L)}$ are the final block's own anchor position/rotation).

No separate final-commit step is needed: there is no distinction between
"what grounded the last block's search" and "what gets committed" — they are
the same anchor, throughout every block.

## 4. Future Work

**Alternative, non-cascaded update rule.** An alternative design was
considered but never implemented: applying only the *last* block's
freshly-computed delta directly to $G_{t-1}$, treating blocks $1..L-1$'s
updates as search-steering only (discarded, not committed). This reflects a
different hypothesis about what iterative refinement is for — "only refine
where to look; decide the total motion once, at the end," rather than this
design's own "converge the position across blocks, and the converged position
is the answer." This alternative would require an additional signal at the
final block (an explicit, un-drifted reference to $G_{t-1}$'s own original
anchor state) to correctly ground its final delta, since the block that
grounds the query's search would otherwise have already drifted from
$G_{t-1}$. This was never empirically validated, since the cascaded design
adopted here required no such additional signal and was assessed as
lower-risk.

**Multi-step training.** As discussed in the paper's own Discussion section,
extending training to a multi-step, sequential setting — where the model
deforms its own previous prediction rather than always starting fresh from a
pristine, ground-truth-derived starting point — is a concrete, promising
direction suggested directly by the ablation study's findings.