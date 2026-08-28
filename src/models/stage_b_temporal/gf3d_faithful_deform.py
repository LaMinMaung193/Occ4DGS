"""
src/models/stage_b_temporal/gf3d_faithful_deform.py

GF3D-faithful Stage B deformation module (Design B, cascaded), implementing
docs/STAGE_B_GF3D_FAITHFUL_DESIGN.md Sections 3.4-3.5.

Reuses GaussianFormer3D's own real DeformableFeatureAggregation3D, AnchorEncoder
(SparseGaussian3DEncoder), FFN, and LayerNorm directly via the same registry-based
build_from_cfg(cfg, MODELS) mechanism GaussianOccEncoder3D itself uses internally
(confirmed against real source: model/encoder/gaussian_encoder/gaussian_encoder_3d.py) --
not hand-instantiated, to avoid guessing at constructor details.

Confirmed per-op instantiation pattern (same file): AnchorEncoder is a SINGLE, shared
instance, reused for every anchor_embed recomputation across all blocks. deformable/ffn/
norm each get INDEPENDENT weights per block (a fresh build() call per block -- this is
simply what happens when build() is called once per list-comprehension entry, not a
special separate mechanism). Our own DeformHeadMu/DeformHeadR follow the same
independent-per-block convention, replacing GF3D's own per-block "refine" op.

Anchor tensor format (28-dim, confirmed byte-precise against the real
SparseGaussian3DEncoder.forward()): [0:3]=mu, [3:6]=scale, [6:10]=rotation,
[10:11]=opacity, [11:28]=semantics.

DeformHeadMu/DeformHeadR both confirmed to take a single concatenated tensor z
(not separate args), and DeformHeadR's forward() already returns a real
quaternion internally (axis_angle_to_quat applied before returning) -- both
verified directly against their real source, not assumed.
"""
import torch
import torch.nn as nn

from src.datasets.gf3d_pipeline import GF3D_ROOT  # noqa: F401 -- triggers sys.path/chdir
from mmengine.registry import build_from_cfg
from mmdet3d.registry import MODELS

from .deform_heads import DeformHeadMu, DeformHeadR, quat_multiply, quat_normalize
from .buffer import GaussianState


# Anchor tensor slice indices -- confirmed against the real SparseGaussian3DEncoder
MU_SLICE = slice(0, 3)
SCALE_SLICE = slice(3, 6)
ROT_SLICE = slice(6, 10)
OPACITY_SLICE = slice(10, 11)
SEMANTIC_SLICE = slice(11, 28)


def gaussian_state_to_anchor(state: GaussianState) -> torch.Tensor:
    """Converts our own GaussianState (buffer format) into the flat 28-dim anchor
    tensor GaussianFormer3D's real modules expect. Assumes state's tensors are
    unbatched (N, ...); returns (1, N, 28) to match GF3D's real (B, N, 28) convention.
    """
    anchor = torch.cat([
        state.means,       # (N, 3)
        state.scales,      # (N, 3)
        state.rotations,   # (N, 4)
        state.opacities,   # (N, 1)
        state.semantics,   # (N, semantic_dim)
    ], dim=-1)
    return anchor.unsqueeze(0)  # (1, N, 28)


def anchor_to_gaussian_state(anchor: torch.Tensor) -> GaussianState:
    """Inverse of gaussian_state_to_anchor. anchor: (1, N, 28) or (N, 28) --
    squeezes the batch dim if present, since GaussianState itself is unbatched."""
    if anchor.dim() == 3:
        anchor = anchor.squeeze(0)
    return GaussianState(
        means=anchor[..., MU_SLICE],
        scales=anchor[..., SCALE_SLICE],
        rotations=anchor[..., ROT_SLICE],
        opacities=anchor[..., OPACITY_SLICE],
        semantics=anchor[..., SEMANTIC_SLICE],
    )


class GF3DFaithfulDeform(nn.Module):
    """Design B (cascaded) deformation module. G_t = anchor^(L), directly --
    no separate final-commit step (design doc Section 3.5)."""

    def __init__(
        self,
        num_blocks: int,
        embed_dims: int,
        num_anchor: int,
        anchor_encoder_cfg: dict,
        deformable_model_cfg: dict,
        norm_cfg: dict,
        ffn_cfg: dict,
        deform_head_hidden_dim: int = 128,
        max_disp_xyz=(4.0, 4.0, 1.0),
        max_angle_rad: float = 0.3,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.embed_dims = embed_dims

        # Single, shared instance -- confirmed against real GaussianOccEncoder3D
        # (self.anchor_encoder built once, reused for every recomputation).
        self.anchor_encoder = build_from_cfg(anchor_encoder_cfg, MODELS)

        # Independent weights per block -- confirmed via GF3D's own op_config_map
        # mechanism (a fresh build() call per list entry).
        #
        # [v2 fix] Real per-block sequence is deformable -> ffn -> norm, ONE
        # norm, positioned after ffn -- confirmed against this exact config's
        # real operation_order. An earlier version of this module had two norm
        # layers (before and after ffn); that was wrong, traced to the design
        # doc's own v2/v3 math which incorrectly stated "two LayerNorms
        # bookending FFN" and was never re-verified against real source until
        # this fix. Also [v2 fix]: our real config's residual_mode="cat", not
        # "add" -- the deformable step's real output is (B,N,2*embed_dims),
        # which is what FFN's in_channels=256 is actually for; FFN reduces
        # back to embed_dims in the same step, not the deformable module itself.
        self.deformable_layers = nn.ModuleList([
            build_from_cfg(deformable_model_cfg, MODELS) for _ in range(num_blocks)
        ])
        self.ffn_layers = nn.ModuleList([
            build_from_cfg(ffn_cfg, MODELS) for _ in range(num_blocks)
        ])
        self.norm_layers = nn.ModuleList([
            build_from_cfg(norm_cfg, MODELS) for _ in range(num_blocks)
        ])

        # Our own heads, independent weights per block (confirmed with the user --
        # matches GF3D's own convention for the "refine" op these heads replace).
        # in_dim = 2 * embed_dims: heads take concat([Q, anchor_embed]), confirmed
        # against their real forward(self, z) signature (single tensor, not two
        # separate args) -- matches our own design doc's stated two-input scope
        # for these heads (Q, anchor_embed), not GF3D's full three-input refine
        # signature (instance_feature, anchor, anchor_embed) -- our heads never
        # need the raw, un-embedded anchor, since they only ever predict
        # delta_mu/delta_r; the frozen properties (scale/opacity/semantics) are
        # copied directly, never routed through these heads at all.
        self.phi_mu_layers = nn.ModuleList([
            DeformHeadMu(in_dim=2 * embed_dims, hidden_dim=deform_head_hidden_dim,
                         max_disp_xyz=max_disp_xyz)
            for _ in range(num_blocks)
        ])
        self.phi_r_layers = nn.ModuleList([
            DeformHeadR(in_dim=2 * embed_dims, hidden_dim=deform_head_hidden_dim,
                       max_angle_rad=max_angle_rad)
            for _ in range(num_blocks)
        ])

        # Q^(0): fixed, learned, per-slot -- independent of anchor's values,
        # matching GaussianFormer3D's own InstanceFeatureEmbedding convention
        # (design doc Section 3.2). This is Stage B's OWN parameter, separate
        # from Stage A's own instance_feature embedding -- a fresh, independently
        # learned table, not reused from Stage A.
        #
        # num_anchor is REQUIRED explicitly (not inferred lazily on first
        # forward()) -- a lazy-init pattern here would silently exclude this
        # parameter from an optimizer built via Adam(model.parameters()) BEFORE
        # the first forward() call, a completely standard training pattern.
        # Gradients would still compute for it, but optimizer.step() would never
        # actually update it -- frozen at random init forever, no error, no
        # crash, nothing visibly wrong. Requiring num_anchor explicitly at
        # construction avoids this entirely.
        self.q0_table = nn.Parameter(torch.randn(num_anchor, embed_dims) * 0.02)

    def forward(
        self,
        g_prev: GaussianState,
        feature_maps,
        dpt_feature_maps,
        metas,
    ) -> GaussianState:
        """
        g_prev: G_{t-1}, read from the reference buffer. Its mu/r are assumed
            ALREADY frame-transformed for projection purposes (via
            transform_anchor_for_projection, Section 3.3) by the caller --
            this module does not do that transform itself, to keep it focused
            on the per-block iteration alone.
        feature_maps, dpt_feature_maps, metas: frame t's real camera/LiDAR
            features, same convention as GaussianFormer3D's own forward().

        Returns: G_t (GaussianState), via G_t = anchor^(L) directly
            (design doc Section 3.5 -- no separate final-commit step under
            Design B).
        """
        anchor = gaussian_state_to_anchor(g_prev)  # (1, N, 28)

        Q = self.q0_table.unsqueeze(0)  # (1, N, embed_dims)
        anchor_embed = self.anchor_encoder(anchor)

        for l in range(self.num_blocks):
            # [v2 fix] deformable's real output is (B,N,2*embed_dims) under our
            # real config's residual_mode="cat" -- Q_cat, not same-shape Q.
            Q_cat = self.deformable_layers[l](
                Q, anchor, anchor_embed, feature_maps, dpt_feature_maps, metas,
                anchor_encoder=self.anchor_encoder,
            )
            Q = self.ffn_layers[l](Q_cat)   # 2*embed_dims -> embed_dims
            Q = self.norm_layers[l](Q)      # ONE norm, after ffn -- real operation_order

            z = torch.cat([Q, anchor_embed], dim=-1)
            delta_mu = self.phi_mu_layers[l](z)
            delta_r = self.phi_r_layers[l](z)

            # Design B cascaded update -- real, not transient (design doc
            # Section 3.4). Position/rotation updated; scale/opacity/semantics
            # copied through unchanged, never routed through the heads.
            new_mu = anchor[..., MU_SLICE] + delta_mu
            new_r = quat_normalize(quat_multiply(anchor[..., ROT_SLICE], delta_r))
            anchor = torch.cat([
                new_mu, anchor[..., SCALE_SLICE], new_r,
                anchor[..., OPACITY_SLICE], anchor[..., SEMANTIC_SLICE],
            ], dim=-1)

            anchor_embed = self.anchor_encoder(anchor)

        return anchor_to_gaussian_state(anchor)
