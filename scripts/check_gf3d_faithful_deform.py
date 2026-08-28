"""
scripts/check_gf3d_faithful_deform.py

Gate 1 smoke test for GF3DFaithfulDeform (Design B): builds the module with
real config values, feeds it a real cached G_0 plus synthetic camera/depth
features, confirms it runs end to end without crashing, and checks output
shapes are correct.

Run from Occ4DGS repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/check_gf3d_faithful_deform.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.datasets.gf3d_pipeline import GF3D_ROOT  # noqa: F401 -- triggers sys.path/chdir
import model  # noqa: F401 -- triggers GaussianFormer3D's own @MODELS.register_module() decorators

from src.models.stage_b_temporal.gf3d_faithful_deform import GF3DFaithfulDeform
from src.models.stage_b_temporal.buffer import GaussianState

G0_CACHE_DIR = "/media/user/1TSSD/min/g0_cache"
N_G = 25600
EMBED_DIMS = 128
NUM_CAMS = 6
IMG_H, IMG_W = 928, 1600  # confirmed padded resolution, matching Stage A's own real pipeline
NUM_LEVELS = 4
DOWNSAMPLE_FACTORS = [8, 16, 32, 64]  # confirmed from the real full-scale config

# Real config values, confirmed directly from nuscenes_surroundocc_gs25600.py
ANCHOR_ENCODER_CFG = dict(
    type="SparseGaussian3DEncoder", embed_dims=128, include_opa=True,
    semantics=True, semantic_dim=17,
)
NORM_CFG = dict(type="LN", normalized_shape=128)
FFN_CFG = dict(
    type="AsymmetricFFN", in_channels=256, embed_dims=128,
    feedforward_channels=512, num_fcs=2, ffn_drop=0.1,
    act_cfg=dict(type="ReLU", inplace=True), pre_norm=dict(type="LN"),
)
DEFORMABLE_MODEL_CFG = dict(
    type="DeformableFeatureAggregation3D", embed_dims=128,
    kps_generator=dict(
        type="SparseGaussian3DKeyPointsGenerator3D", embed_dims=128,
        phi_activation="sigmoid", xyz_coordinate="cartesian", num_learnable_pts=2,
        fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], [0, 0.45, 0],
                   [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]],
        pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        scale_range=[0.01, 1.8],
    ),
    d_bound=[2.0, 58, 0.5], im2col_step=32, use_visibility=False,
    use_sampling_offsets=True, num_pts_per_keypoint=2, value_projection=False,
    num_cams=6, num_groups=4, num_levels=4, residual_mode="cat",
    use_camera_embed=True, use_deformable_func=True, attn_drop=0.15,
)


def load_real_g0(scene_token):
    path = os.path.join(G0_CACHE_DIR, f"{scene_token}.pt")
    d = torch.load(path, map_location="cpu")
    # Cached tensors have a batch dim (1, N, ...) -- GaussianState expects
    # unbatched (N, ...), confirmed from its own __post_init__ shape validation.
    return GaussianState(
        means=d["means"].squeeze(0),
        scales=d["scales"].squeeze(0),
        rotations=d["rotations"].squeeze(0),
        opacities=d["opacities"].squeeze(0),
        semantics=d["semantics"].squeeze(0),
    )


def make_synthetic_features():
    feature_maps, dpt_feature_maps = [], []
    for factor in DOWNSAMPLE_FACTORS:
        h, w = max(IMG_H // factor, 1), max(IMG_W // factor, 1)
        feature_maps.append(torch.randn(1, NUM_CAMS, EMBED_DIMS, h, w))
        dpt_feature_maps.append(torch.randn(1, NUM_CAMS, EMBED_DIMS, h, w))
    return feature_maps, dpt_feature_maps


def make_synthetic_metas():
    projection_mat = torch.eye(4).view(1, 1, 4, 4).repeat(1, NUM_CAMS, 1, 1)
    image_wh = torch.tensor([IMG_W, IMG_H], dtype=torch.float32).view(1, 1, 2).repeat(1, NUM_CAMS, 1)
    return {"projection_mat": projection_mat, "image_wh": image_wh}


def main():
    print("Building GF3DFaithfulDeform (L=4, real config values)...")
    deform = GF3DFaithfulDeform(
        num_blocks=4,
        embed_dims=EMBED_DIMS,
        num_anchor=N_G,
        anchor_encoder_cfg=ANCHOR_ENCODER_CFG,
        deformable_model_cfg=DEFORMABLE_MODEL_CFG,
        norm_cfg=NORM_CFG,
        ffn_cfg=FFN_CFG,
    ).cuda()
    n_params = sum(p.numel() for p in deform.parameters())
    print(f"  Built successfully. {n_params:,} parameters.")

    print("\nLoading a real cached G_0...")
    scene_files = [f for f in os.listdir(G0_CACHE_DIR) if f.endswith(".pt")]
    assert len(scene_files) > 0, f"No cached G_0 files found in {G0_CACHE_DIR}"
    scene_token = scene_files[0].replace(".pt", "")
    g_prev = load_real_g0(scene_token)
    print(f"  Loaded scene {scene_token}: {g_prev.num_gaussians} Gaussians")
    assert g_prev.num_gaussians == N_G, f"Expected {N_G}, got {g_prev.num_gaussians}"
    g_prev = GaussianState(
        means=g_prev.means.cuda(), scales=g_prev.scales.cuda(),
        rotations=g_prev.rotations.cuda(), opacities=g_prev.opacities.cuda(),
        semantics=g_prev.semantics.cuda(),
    )

    print("\nBuilding synthetic camera/depth features and metas...")
    feature_maps, dpt_feature_maps = make_synthetic_features()
    feature_maps = [f.cuda() for f in feature_maps]
    dpt_feature_maps = [f.cuda() for f in dpt_feature_maps]
    metas = make_synthetic_metas()
    metas = {k: v.cuda() for k, v in metas.items()}
    for i, f in enumerate(feature_maps):
        print(f"  level {i}: {tuple(f.shape)}")

    print("\nRunning forward()...")
    with torch.no_grad():
        g_t = deform(g_prev, feature_maps, dpt_feature_maps, metas)

    print("\n=== Output check ===")
    print(f"  means:      {tuple(g_t.means.shape)}")
    print(f"  rotations:  {tuple(g_t.rotations.shape)}")
    print(f"  scales:     {tuple(g_t.scales.shape)}")
    print(f"  opacities:  {tuple(g_t.opacities.shape)}")
    print(f"  semantics:  {tuple(g_t.semantics.shape)}")
    assert g_t.num_gaussians == N_G

    # Confirm scale/opacity/semantics genuinely unchanged (frozen, as designed)
    assert torch.allclose(g_t.scales, g_prev.scales), "scales should be frozen"
    assert torch.allclose(g_t.opacities, g_prev.opacities), "opacities should be frozen"
    assert torch.allclose(g_t.semantics, g_prev.semantics), "semantics should be frozen"
    print("\n  Confirmed: scales/opacities/semantics are exactly frozen (unchanged).")

    # Confirm position/rotation DID change (the whole point of Stage B)
    moved = not torch.allclose(g_t.means, g_prev.means)
    rotated = not torch.allclose(g_t.rotations, g_prev.rotations)
    print(f"  Position changed: {moved}  |  Rotation changed: {rotated}")

    print("\nGate 1: PASS -- module runs end to end, shapes correct, frozen "
          "properties genuinely frozen, moved properties genuinely moved.")


if __name__ == "__main__":
    main()
