Occ4DGS Stage B: High-Level Design and Math
(t>0 deformation module, reusing GaussianFormer3D's real architecture directly)

v3 -- Design B adopted as the primary implementation (lower-risk, matches
GaussianFormer3D's own within-frame convergence pattern exactly, no open
validation question). Design A (the original v1/v2 choice) is preserved below,
relabeled as a documented, concrete FUTURE ABLATION -- to be tested specifically
if Design B underperforms, not the current default. All v2 source-verified
corrections (kps_generator's real anchor+Q construction, AnchorEncoder's real
five-way split, FFN/LayerNorm ordering, Deformation Heads' real input signature)
are carried forward unchanged -- only the final-commit strategy changes in this
revision.

Companion document to the architecture diagram (Figure 1). Covers: 2) pipeline
flow and per-module input/output; 3) formal math.

---

## 2. High-Level Design: Pipeline, Flow, and Module I/O

### 2.1 Overview

t=0 is entirely unchanged GaussianFormer3D, run once, producing G_0. Every
subsequent frame (t>0) runs a feedforward deformation step that reuses
GaussianFormer3D's own real modules directly -- not reimplementations -- applied
to G_{t-1} (the reference buffer's current Gaussian state) instead of a
freshly-initialized one.

[v3] Final-commit strategy: Design B, cascaded, adopted as primary. Every
block's Deformation Heads output is applied directly to that block's own
running anchor, and the anchor itself -- not a separately-tracked delta --
carries forward into the next block's attention. After the last block (l=L),
G_t = anchor^(L) directly. This matches GaussianFormer3D's own real per-block
behavior exactly: GF3D's own refine step updates the anchor every block and
feeds the updated anchor into the next block's kps_generator, with no separate
"transient vs. final" distinction anywhere in their real code. Design B carries
no open validation question, since there is no mismatch between what grounds
the query's search and what gets committed -- they are the same anchor,
throughout.

Why not Design A (the original choice): Design A applied only the last block's
freshly-computed delta to G_{t-1} directly, discarding blocks 1..L-1's deltas
as search-steering only. This is not a compute-saving choice (both designs run
the identical number of attention calls) -- it is a different hypothesis about
what iterative refinement is for: "converge the position across blocks, and the
converged position is the answer" (Design B) versus "only refine where to look;
decide the total motion once, at the end" (Design A). Design A requires an
additional, not-yet-empirically-validated signal (documented in Section 3.6
below) to bridge the resulting mismatch between what grounded the query's
evidence-gathering and what actually gets committed. Given this adds real
implementation and validation risk with no compute benefit, Design B is
adopted as the primary implementation; Design A is kept as a documented,
ready-to-run future ablation if Design B underperforms.

### 2.2 Module-by-module flow, t>0

Step 1 -- Backbones (reused, unchanged)
- Input: frame t's 6 camera images + LiDAR point cloud.
- Output: F^c_t (multi-scale camera features), F^d_t (multi-scale
  depth-conditioned features) -- same CurrentFrameEncoder already built and
  tested earlier in this project; no new module here.

Step 2 -- Outer Product / KV construction (reused, unchanged)
- Input: F^c_t, F^d_t.
- Output: not a materialized tensor -- confirmed from GF3D's real source that
  F^3D = F^d (x) F^c describes what DeformableFeatureAggregation3D's CUDA
  kernel achieves functionally, not a separate step. F^c_t/F^d_t are passed
  directly into Step 4 as feature_maps/dpt_feature_maps.

Step 3 -- Query and anchor construction

Two tensors feed every block, each with a different job:

- Anchor -- the Gaussian's explicit properties (position, scale, rotation,
  opacity, class; 28 values, all interpretable -- 3 (mu) + 4 (rot) + 3 (scale)
  + 1 (opacity) + 17 (semantic classes) = 28, confirmed against
  SparseGaussian3DEncoder's real index slicing). Comes directly from the
  reference buffer, G_{t-1}, at block 1; [v3] under Design B, evolves as the
  real, cascading Gaussian state block-to-block -- not a transient value
  discarded at the end, but the actual position/rotation the model is
  converging toward G_t itself. Tells the model where the Gaussian currently
  is.
- Q (instance_feature) -- an abstract, high-dimensional working state (not
  interpretable). Starts as a fixed, learned, per-Gaussian-slot value
  (InstanceFeatureEmbedding), independent of the anchor's specific values --
  the same design GaussianFormer3D itself uses. Confirmed against real
  GaussianOccEncoder3D.forward(): this table is random at the start of
  training, shaped by gradient descent over training, and frozen as trained
  values once training finishes; at inference it is the same fixed starting
  point for every scene, with each scene's own information entering only
  afterward via attention. Decides how much to trust, and where exactly, to
  sample.
- Combining them -- before each block's attention, the anchor is embedded into
  the same high-dimensional space as Q (anchor_embed = AnchorEncoder(anchor)),
  then added directly to Q. AnchorEncoder's real internal structure
  (SparseGaussian3DEncoder): it does not project the flat 28-dim vector in one
  pass -- it splits the anchor into five separate property groups (position,
  scale, rotation, opacity, class), encodes each independently into embed_dims
  via its own small linear_relu_ln stack, sums the five results, then applies
  one final output projection. This is how the Gaussian's real, current
  properties reach the attention step, even though Q itself didn't start from
  those values.

Step 4 -- 3D Deformable Attention (reused verbatim -- DeformableFeatureAggregation3D)
- Input: instance_feature (Q), anchor, anchor_embed, feature_maps (F^c_t),
  dpt_feature_maps (F^d_t), metas (incl. projection_mat, image_wh).
- Internally, confirmed line-by-line against deformable_module_3d.py:
  1. key_points = self.kps_generator(anchor, instance_feature) -- reference key
     points are anchor-dominant but not anchor-only. kps_generator
     (SparseGaussian3DKeyPointsGenerator3D) builds a fixed geometric template
     from the anchor's real scale/rotation/position, and, when
     num_learnable_pts > 0, concatenates an additional learnable-scale
     template derived from instance_feature via self.learnable_fc. So key
     points are anchor-dominant but not exclusively anchor-derived.
  2. points_3d, bev_mask = self.project_points_3d(key_points, projection_mat,
     d_bound, image_wh) -- pure geometric projection into normalized (u, v,
     depth) space; depth is preserved as a real coordinate (confirmed: this is
     the exact code-level fix for the 2D version's depth-ambiguity problem --
     the 2D module's project_points() computes depth only to divide by it,
     then discards it, keeping just (u, v)).
  3. weights = self._get_weights_3d(instance_feature, anchor_embed, metas) --
     computed from feature = instance_feature + anchor_embed, through
     weights_fc, softmaxed per-camera over (levels x pts).
  4. If use_sampling_offsets: the same feature = instance_feature +
     anchor_embed is passed through learned linear layers (sampling_offsets,
     sampling_offsets_depth) to produce delta-u, delta-v, delta-depth, added
     directly onto the already-projected points_3d (in normalized (u,v,depth)
     space, not raw 3D world space) -- this is the two-stage sampling
     refinement (Stage 1: anchor/Q-derived geometric reference points; Stage
     2: Q+anchor_embed-derived learned offsets around each of those points).
  5. MultiScaleDeformableAttnFunction.apply(...) -- the low-level CUDA kernel:
     samples the fused feature volume at the final points, multiplies by
     weights, sums. This is the only part of the pipeline that is a literal
     "read the feature map, weight it, sum" operation -- everything
     determining where to sample and how much to trust each sample happens
     beforehand, inside this same module, using Q/anchor/anchor_embed as real
     inputs.
  6. output = self.output_proj(slots); [v5, CORRECTED] our real config uses
     residual_mode="cat", not "add" -- confirmed directly from
     deformable_module_3d.py's real source. output = torch.cat([output,
     instance_feature], dim=-1), giving (B, N, 2*embed_dims), NOT a same-shape
     residual add. This concatenation happens inside this module, before it
     returns -- explains why the real FFN config has in_channels=256 (2x
     embed_dims=128): the FFN is what reduces back to embed_dims, not this
     module's own output.
- Output: Q_new -- same shape as Q ((B, N_g, embed_dims)).

Step 5 -- FFN + LayerNorm (reused, unchanged)
- Input: Q_new (the (B,N,256) concatenated output from Step 4, under our real
  residual_mode="cat").
- Flow, [v5, CORRECTED] confirmed against this exact config's real
  operation_order (['deformable','ffn','norm','refine', 'spconv','norm',
  'deformable','ffn','norm','refine', ...]): Q_new -> FFN -> LayerNorm -> Q^(l)
  -- ONE LayerNorm, positioned AFTER FFN, not two bookending it (this was
  incorrectly stated as "two LayerNorms" in v2/v3 and never caught until this
  revision). FFN reduces 256 -> 128 in the same step. Note: GF3D's own blocks
  2-4 also have a norm right after spconv, before deformable -- but that norm
  exists specifically to stabilize spconv's own output, which our design
  doesn't use at all (disclosed simplification, all blocks). Every block in
  OUR design uses the same, uniform, spconv-free sequence (matching GF3D's own
  block-1 pattern), not GF3D's block-1-vs-later distinction.
- Role: attention only produces a weighted sum of sampled values -- a linear
  combination; it can only interpolate between what it's given, never create a
  genuinely new pattern. FFN adds non-linear processing on top, independently
  per Gaussian, letting the model transform what attention gathered more
  expressively than a weighted sum alone allows. Each LayerNorm keeps Q's
  values from drifting to extreme magnitudes after the residual addition that
  precedes it -- the same category of fix this project already had to build
  and prove necessary in an earlier design (after a real training-instability
  bug); here it comes for free from GF3D's own proven block.
- Output: Q^(l) -- the value the rest of the pipeline treats as "the current
  Q," finalized for this block.

Step 6 -- Deformation Heads (our own, unchanged from every prior design this
project has built)
- Input: Q^(l), anchor_embed^(l-1) -- matching GF3D's own real refine_module.py
  signature (instance_feature, anchor, anchor_embed), not Q alone.
- Output: Delta_mu_i^(l), Delta_r_i^(l) -- via Phi_mu, Phi_r, same residual,
  tanh-bounded heads used throughout this project.

Step 7 -- Anchor update, every block (Design B -- real, not transient)
- The anchor's position and rotation are updated directly by this block's
  Delta_mu/Delta_r; scale/opacity/semantics remain frozen (unchanged from
  every prior design this project has built -- these properties are
  time-invariant under Stage B's update rule).
- anchor_embed is recomputed fresh from the updated anchor, ready for the next
  block's attention.
- [v3] Under Design B, this update is real and cumulative -- not a transient
  value used only to seed search and then discarded. After block L, the
  anchor itself, in its current state, becomes G_t directly (Section 3.4).

---

## 3. Math Design

(Sections 3.1-3.3 -- notation, query/anchor initialization, and the frame-
transform-for-projection question -- unchanged from v2; carried forward below
for completeness.)

### 3.1 Setup

    G_{t-1} = {mu_i, r_i, s_i, alpha_i, c_i}, i = 1..N_g   (reference buffer)
    anchor^(0)_i = concat(mu_i, s_i, r_i, alpha_i, c_i)     (28-dim, per Gaussian)

### 3.2 Query and anchor initialization

    Q^(0) = InstanceFeatureEmbedding      -- fixed, learned, per-slot; independent of anchor
    anchor_embed^(0) = AnchorEncoder(anchor^(0))

### 3.3 Frame transform before projection -- RESOLVED (v4)

anchor^(0)'s position (mu_i) AND rotation (r_i) are stored in G_{t-1}'s own
frame. GaussianFormer3D's real project_points_3d (used inside Step 4, via
kps_generator) requires positions already in frame t's local LiDAR frame,
matching frame t's projection_mat -- confirmed necessary in this project's
earlier work (Option D design, the professor Q&A on ego-motion/coordinate-frame
necessity). kps_generator ALSO uses the anchor's rotation to orient its
sampling template (an oriented ellipsoid, not a sphere) -- so the same
frame-consistency requirement applies to rotation, not just position: if the
ego vehicle turned between t-1 and t, "up/forward/right" in frame t-1's terms
is not the same as in frame t's terms. Leaving rotation unconverted while
converting only position would build an internally inconsistent template.

Before the anchor is passed into the reused DeformableFeatureAggregation3D, a
transient (not persisted) copy of BOTH its position and rotation must be
transformed:

    T = compute_relative_transform(pose_prev, pose_curr)   -- (4,4), rotation
                                                              included, confirmed
                                                              directly from the
                                                              real code: T = inv
                                                              (pose_curr) @ pose_prev,
                                                              both full SE(3) poses
    mu_i^(proj) = T @ [mu_i, 1]^T
    q_rel = rotmat_to_quat(T[:3, :3])
    r_i^(proj) = quat_multiply(q_rel, r_i)

Resolution confirmed: compute_relative_transform was never translation-only --
it already returns a full rigid transform (verified directly against the real
function in deform_heads.py), so no new utility function is needed.
rotmat_to_quat and quat_multiply both already exist in the same file. The gap
was only ever in this design doc's own math, which previously applied T to
mu_i but never took T's rotational part and applied it to r_i. That gap is
fixed above.

Only the transient projection-input copy is affected either way -- G_{t-1}'s
actual stored mu_i, r_i are never reassigned to a new frame; this mirrors
Option D's established convention (transform for projection purposes only, not
persistent ego-compensation, which was tested separately and found
net-negative when combined with learning).

### 3.4 Per-block iteration, l = 1..L [v3 -- Design B, cascaded]

    Q_cat^(l) = DeformableFeatureAggregation3D(Q^(l-1), anchor^(l-1), anchor_embed^(l-1),
                                                 F^c_t, F^d_t, metas)
             -- [v5, CORRECTED] residual_mode="cat" is our real config's actual
             -- value (confirmed directly from deformable_module_3d.py source,
             -- NOT "add" as v2/v3 assumed): output = concat(output_proj(slots),
             -- Q^(l-1)), giving (B, N, 2*embed_dims), not a same-shape residual add.

    Q^(l) = FFN(Q_cat^(l))           -- [v5] reduces 2*embed_dims -> embed_dims;
                                      -- matches the real operation_order
                                      -- (deformable -> ffn -> norm, confirmed NO
                                      -- norm between deformable and ffn)
    Q^(l) = LayerNorm(Q^(l))         -- [v5] ONE norm per block, after FFN --
                                      -- GF3D's norm-after-spconv (blocks 2-4)
                                      -- doesn't apply to us (no spconv, any block)

    Delta_mu_i^(l) = Phi_mu(Q_i^(l), anchor_embed_i^(l-1))
    Delta_r_i^(l)  = Phi_r(Q_i^(l), anchor_embed_i^(l-1))

    -- [v3] Real, cascading update -- every block, no transient/final
    -- distinction. anchor^(l) directly carries forward into block l+1's
    -- attention, exactly matching GF3D's own real per-block behavior.
    anchor^(l)_mu = anchor^(l-1)_mu + Delta_mu_i^(l)
    anchor^(l)_r  = normalize(quat_multiply(anchor^(l-1)_r, Delta_r_i^(l)))   -- current-first, Step 1's fix
    anchor^(l)_s, anchor^(l)_alpha, anchor^(l)_c = anchor^(l-1)_s, anchor^(l-1)_alpha, anchor^(l-1)_c   -- frozen, as decided
    anchor_embed^(l) = AnchorEncoder(anchor^(l))

### 3.5 Final update [v3 -- Design B: G_t is the last block's own anchor, directly]

    G_t = anchor^(L)

That is: mu_{t,i} = anchor^(L)_mu_i, r_{t,i} = anchor^(L)_r_i,
s_{t,i} = s_{t-1,i}, alpha_{t,i} = alpha_{t-1,i}, c_{t,i} = c_{t-1,i}.

No separate final-commit step is needed under Design B -- unlike Design A
(Section 3.6 below), there is no distinction between "what grounded the last
block's search" and "what gets committed": they are the same anchor,
throughout every block. This is precisely why Design B carries no open
validation question that Design A required.

### 3.6 [Preserved for a future ablation -- NOT part of the current, active
implementation] Design A: non-cascaded, single final commit

This section documents the original (v1/v2) design choice, kept here as a
ready-to-run, concrete future ablation if Design B underperforms -- not the
current default.

Under Design A, only the last block's freshly-computed delta is applied,
directly to G_{t-1} (not to the accumulated anchor^(L-1)):

    mu_{t,i} = mu_{t-1,i} + Delta_mu_i^(L)
    r_{t,i}  = normalize(quat_multiply(r_{t-1,i}, Delta_r_i^(L)))

Blocks 1..L-1's anchor updates, under Design A, are transient -- used only to
seed where the next block's kps_generator searches, then discarded, never
numerically added into the final Gaussian.

Design A's own required fix (block-L conditioning): because
Delta_mu^(L)/Delta_r^(L) are committed to G_{t-1}, not to anchor^(L-1) (the
anchor that actually grounded Q^(L)'s attention), a delta naturally computed
relative to anchor^(L-1) is not automatically correct relative to G_{t-1},
since anchor^(L-1) has drifted from G_{t-1} by the (discarded, under Design A)
accumulated sum of blocks 1..L-1's deltas. The proposed fix: inject an
additional, explicit signal at block L only --

    anchor_embed_G0_i = AnchorEncoder(anchor^(0)_i)   -- G_{t-1}'s original anchor,
                                                        -- recomputed fresh at block L

    Delta_mu_i^(L) = Phi_mu(Q_i^(L), anchor_embed_i^(L-1), anchor_embed_G0_i)
    Delta_r_i^(L)  = Phi_r(Q_i^(L), anchor_embed_i^(L-1), anchor_embed_G0_i)

anchor_embed^(L-1) must still be used to keep Q^(L)'s grounding consistent
with what generated it (do not replace it with anchor_embed_G0 -- this was
considered and rejected, since it would create a mismatch between what Q^(L)
"knows" and what conditions the residual heads). anchor_embed_G0 is added
alongside it, purely so the heads have an explicit, un-drifted reference point
for what they are actually being asked to predict relative to.

Status: proposed fix, never empirically validated -- Design A was superseded
by Design B before implementation began. If this ablation is run later, this
fix should be implemented and tested as described here, and its necessity
confirmed (or refuted) empirically, not assumed.

---

## Open items before implementation

1. Section 3.3's rotation-transform question -- needs a deliberate decision,
   flagged explicitly, not assumed. Unresolved since v1.
2. Zero-valid-camera fallback behavior, K/offset-count defaults -- carried
   from the prior VGGT-based design, still applicable.
3. Per-block independent weights -- confirmed already, via
   GaussianOccEncoder3D's real nn.ModuleList construction (every op in every
   block gets its own instance, nothing shared).
4. [If Design A is ever tested later] Section 3.6's block-L extra
   conditioning signal needs empirical validation (e.g., ablate with/without
   anchor_embed_G0 at block L) -- not an open item for the current, Design-B
   implementation.
5. Worth a quick source check, not yet done: confirm in the real training
   config whether mid_refine_layer differs from refine_layer in
   GaussianOccEncoder3D -- if so, worth understanding as a point of comparison
   against this project's own Design A/B distinction, even though Design B is
   now the active choice.

---

## Changelog

v3 (this revision): Design B (cascaded, matching GF3D's own within-frame
convergence exactly) adopted as the primary implementation, replacing Design A
as the default. Reasoning: both designs cost the same (identical number of
attention calls); Design A required an additional, never-validated signal
(Section 3.6) to bridge a mismatch Design B does not have in the first place.
Design A is fully preserved, relabeled as a documented, ready-to-run future
ablation (Section 3.6), not deleted. Sections 3.4/3.5 rewritten to reflect
Design B's simpler, cascaded update rule; Section 2.1's framing corrected
(the original "for lighter compute" justification for Design A did not
actually hold up -- both designs have identical compute cost, confirmed).

v2 (prior revision): updated after direct source review of
deformable_module_3d.py, deformable_module.py, anchor_encoder_module.py,
gaussian_encoder_3d.py, and refine_module.py:

1. kps_generator corrected: reference key points are anchor-dominant but not
   anchor-only -- instance_feature (Q) also contributes a learnable-scale
   template when num_learnable_pts > 0.
2. AnchorEncoder's real structure documented: five separate per-property
   encoders summed, not one flat-vector MLP.
3. FFN+LayerNorm order corrected: LayerNorm -> FFN -> LayerNorm, not a single
   LayerNorm after FFN.
4. Anchor/anchor_embed evolution confirmed exactly: reassigned after every
   block, always the immediately preceding block's values -- not a fixed
   block-0 value, confirmed directly against GaussianOccEncoder3D's real
   variable reuse.
5. mid_refine precedent noted: GF3D's own code already has a distinct
   mid_refine op separate from refine -- worth checking as a precedent (still
   an open item, see Open Items #5 above).
6. Deformation Heads' input signature clarified: (Q, anchor, anchor_embed),
   matching GF3D's real refine_module.py signature exactly -- not Q alone.
7. Design A vs. Design B explicitly decided and documented (later superseded
   by v3 above).
8. Step 7a / Section 3.6 added: block L's Deformation Heads need an explicit
   anchor_embed(G_{t-1}) signal under Design A (later made moot by the v3
   switch to Design B, preserved as future-ablation documentation).
9. refine_module.py's real behavior confirmed: it already performs
   per-property residual updates (refine_state-gated), not a from-scratch
   prediction of the whole Gaussian -- direct precedent for treating anchor as
   a required, not optional, input to any refinement/deformation head.
