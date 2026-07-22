"""
src/models/stage_b_temporal/current_frame_encoder.py

Phase 5: standalone per-frame encoder for Stage B's temporal unroll. Wraps
BEVSegmentorLiDAR3D's img_backbone / img_neck / pts_dpt_head directly (frozen, reused
verbatim per design_doc_v2.md Section 1.2 / Section 2.2), replicating the exact call
order and tensor reshaping in GaussianFormer3D's own extract_img_dpt_feat() /
extract_multiscale_dpt() (model/segmentor/bev_segmentor_lidar_3d.py) -- but calling the
three submodules directly rather than calling those two methods verbatim.

Confirmed against source (EXPERIMENT_LOG.md, Phase 5 bridge investigation):
  - For B=1 (Phase 5's batch size, per to_batch_of_one()), extract_img_dpt_feat's
    `imgs.squeeze_(0)` branch and its B>1 `.reshape(B*N,...)` branch produce the same
    (B*N, C, H, W) shape -- this wrapper always uses .reshape(), which is correct for
    both cases, not just B=1.
  - pts_dpt_head is called with the *reshaped* (B, N, C_embed, H_l, W_l) `ms_img_feats`
    as its first argument, but the *flat* (B*N, C_dpt, H, W) `dpt_masked` as its third --
    two different shape conventions in the same call, confirmed by reading
    extract_img_dpt_feat's return dict (`ms_img_feats` already reshaped, `dpt_masked`
    left flat) feeding directly into extract_multiscale_dpt's pts_dpt_head call.
  - lidar2img (== metas["projection_mat"]) is passed through .flatten(2), i.e.
    (B, N, 4, 4) -> (B, N, 16), matching extract_multiscale_dpt exactly.
  - out_dpt_multiscale's final reshape uses lidar2img.shape[:2] (i.e. (B, N)) as the
    unflatten target, not a recomputation from img_feats' own shape.

Two deliberate differences from the original methods, both logged here rather than
silently diverging:

1. GridMask augmentation is skipped entirely (not just eval()'d). Confirmed that
   GridMaskHybrid.forward() is gated by self.training internally, so putting the whole
   model in .eval() would technically suffice -- but Stage B's encoder call happens
   inside an outer training loop where the temporal module (HyperNet, deform heads)
   needs .train()-mode behavior, while these frozen encoders should not receive random
   augmentation. Since .training is a single flag per submodule tree shared with the
   borrowed submodules, it's simpler and unambiguous to never call grid_mask here at all,
   rather than rely on toggling .eval()/.train() correctly on exactly the right submodule
   at exactly the right time. Frozen Stage-A encoder features must be deterministic
   across the unroll, since Stage B's whole premise is inferring real inter-frame motion
   from feature differences -- injected per-step random masking would be an unwanted
   confound, not useful regularization for a frozen encoder.

2. extract_lidar_feat (voxelize + lidar_voxel_encoder) is not called at all. Confirmed
   (GaussianHead.forward() reads only `gaussian` dict properties, never voxel_lidar_feats
   directly) that voxel_lidar_feats is only consumed by `lifter` (one-time Gaussian
   initialization at frame 0, i.e. Stage A proper) -- Stage B's per-frame step never
   touches lifter/encoder/the iterative refinement decoder (GaussianHead is callable
   standalone -- see EXPERIMENT_LOG.md Phase 5 bridge finding #3), so nothing here needs
   raw LiDAR points at all.

Input shapes match to_batch_of_one()'s output (scripts/run_stage_a_frame0.py) exactly:
    imgs: (B, N_cam=6, C=3, H, W) float32, already normalized + padded by the pipeline
    dpt:  (B, N_cam=6, 1, H, W) float32, or None
    metas["projection_mat"]: (B, N_cam=6, 4, 4) float32
"""


class CurrentFrameEncoder:
    """
    Stateless wrapper around a loaded BEVSegmentorLiDAR3D instance's frozen submodules.
    Does not own, freeze, or .eval() the model -- the caller is responsible for
    `requires_grad_(False)` and eval-mode on img_backbone/img_neck/pts_dpt_head
    themselves; this class only controls call order and bypasses grid_mask, it does not
    set grad or train/eval mode on anything.
    """

    def __init__(self, segmentor):
        self.img_backbone = segmentor.img_backbone
        self.img_neck = segmentor.img_neck
        self.pts_dpt_head = segmentor.pts_dpt_head
        # img_backbone_out_indices lives on the segmentor itself (see
        # bev_segmentor_lidar_3d.py __init__), not on img_backbone -- reused verbatim
        # rather than re-derived/guessed, since it selects which backbone stages feed FPN.
        self.img_backbone_out_indices = segmentor.img_backbone_out_indices

    def encode(self, imgs, dpt, metas):
        """
        Args:
            imgs: (B, N, C, H, W) float tensor, already normalized/padded.
            dpt:  (B, N, 1, H, W) float tensor, or None.
            metas: dict containing at least "projection_mat": (B, N, 4, 4).

        Returns:
            ms_img_feats: list[Tensor], one per FPN level, each
                          (B, N, C_embed, H_l, W_l).
            dpt_dist: Tensor, pts_dpt_head's raw depth-distribution output (not
                      per-camera reshaped -- shape is whatever DepthHead_GTDpt itself
                      returns; not consumed by Stage B directly today, returned for
                      completeness/future logging).
            out_dpt_multiscale: list[Tensor], one per level, each
                      (B, N, C_dpt_embed, H_l, W_l) -- reshaped identically to
                      extract_multiscale_dpt()'s own out_dpt_multiscale construction.
        """
        B, N, C, H, W = imgs.shape
        imgs_flat = imgs.reshape(B * N, C, H, W)

        dpt_flat = None
        if dpt is not None:
            _, _, C_dpt, H_dpt, W_dpt = dpt.shape
            dpt_flat = dpt.reshape(B * N, C_dpt, H_dpt, W_dpt)
        # NOTE: no grid_mask call here -- see module docstring point 1. dpt_flat stands
        # in for extract_img_dpt_feat's `dpt_masked` return value, minus the masking.

        img_feats_backbone = self.img_backbone(imgs_flat)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats_selected = [img_feats_backbone[idx] for idx in self.img_backbone_out_indices]
        img_feats = self.img_neck(img_feats_selected)  # list[Tensor], each (B*N, C_embed, H_l, W_l)

        ms_img_feats = []
        for feat in img_feats:
            BN, C_embed, H_l, W_l = feat.shape
            assert BN == B * N, (
                f"CurrentFrameEncoder: img_neck output batch dim {BN} != B*N ({B * N}) "
                f"-- a reshape assumption here is wrong, do not silently proceed"
            )
            ms_img_feats.append(feat.view(B, N, C_embed, H_l, W_l))

        projection_mat = metas["projection_mat"]  # (B, N, 4, 4)
        # NOTE: pts_dpt_head takes the RESHAPED ms_img_feats but the FLAT dpt_flat --
        # this asymmetry is confirmed against source, not a typo. See module docstring.
        dpt_dist, out_dpt_multiscale_raw = self.pts_dpt_head(
            ms_img_feats, projection_mat.flatten(2), dpt_flat, return_dpt=True
        )
        out_dpt_multiscale = [
            outdpt.view(*projection_mat.shape[:2], *outdpt.shape[1:])
            for outdpt in out_dpt_multiscale_raw
        ]

        return ms_img_feats, dpt_dist, out_dpt_multiscale