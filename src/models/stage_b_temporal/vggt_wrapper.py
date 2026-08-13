"""
src/models/stage_b_temporal/vggt_wrapper.py

VGGT_DEFORMABLE_DESIGN.md Section 7, item 1. Thin wrapper exposing ONLY VGGT's dense
per-camera-per-frame feature tokens -- never its own camera/depth/point/track
predictions (Section 2: "VGGT's role: pure dense feature extractor... never read out
or rely on VGGT's own predicted camera parameters or point maps"). Geometric
anchoring for sampling these features uses our own real, calibrated projection_mat
(reused from Option D), not anything VGGT itself believes about the scene.

Frozen (Section 2: "VGGT trainable? Frozen").

Open items 1 & 2 from the design doc, both confirmed against the real
facebookresearch/vggt source (cloned and inspected, not assumed) before writing this
file:

1. Feature extraction API: `Aggregator(images)` (images: (B, S, 3, H, W)) returns
   `(output_list, patch_start_idx)`. `output_list` has one entry per transformer
   block (24 for VGGT-1B), None everywhere except at `cached_layer_indices`
   (default (4, 11, 17, 23)), where entries are (B, S, P, 2*embed_dim) -- the
   frame-attention and global-attention streams concatenated (confirmed: VGGT's own
   CameraHead/DPTHead/TrackHead all consume this same `dim_in=2*embed_dim`
   convention). `patch_start_idx = 1 + num_register_tokens` (camera token + register
   tokens prepended before the real patch tokens) -- verified by executing a small
   random-init Aggregator (patch_embed="conv", cheap) end-to-end and inspecting the
   real output shapes directly, not just reading source.
2. Channel dimension: 2*embed_dim (2048 for the real VGGT-1B checkpoint,
   embed_dim=1024) -- but this module does NOT hardcode that number. It infers the
   true channel dim at construction time via one tiny dummy forward pass, so it stays
   correct regardless of which checkpoint/config is actually loaded.

This wrapper uses ONLY the final cached layer (the deepest, most semantically
abstracted one) -- Liam's confirmed decision: start simple, consistent with the
discipline that's paid off all session (Step 4's smaller network matching a larger
one, Step 5's added complexity hurting). Explicit, honest caveat carried over from
that decision, not silently dropped: this trades away exactly the fine spatial detail
a task built on sampling precise projected pixel locations may need most -- full
4-layer DPT-style multi-scale fusion is the specific, well-motivated next step if
Gate 2/3 shows the mechanism working but underperforming on spatial precision, not a
vague someday-maybe.

Both frames are fed to VGGT TOGETHER in one sequence (Section 2: "VGGT input: both
frames fed together... uses VGGT's native multi-frame alternating-attention"), so its
own frame-wise/global attention does real multi-frame reasoning internally -- this is
why feat_curr can show some influence from images_prev's content (global attention
mixes across the whole sequence) and vice versa; that's correct VGGT behavior, not a
leak this wrapper should try to prevent. See the standalone test for what IS checked:
the prev/curr split and the per-camera indexing are correctly aligned, which is what
this wrapper's OWN code (not VGGT's internals) could get wrong.
"""
import torch
import torch.nn as nn

# Side effect only: puts VGGT's own repo root on sys.path, matching how
# src/datasets/gf3d_pipeline.py does the same for GF3D_ROOT. Real bug found on the
# lab machine (Gate 1's ModuleNotFoundError: No module named 'vggt') -- this file's
# real (non-test-injection) __init__ path does `from vggt.models.vggt import VGGT`
# with nothing, until now, ensuring `vggt` was actually importable; the test file
# had its own local sys.path setup, but that never covered the production path.
import os
import sys

VGGT_ROOT = os.path.expanduser("~/Documents/min/vggt")
if VGGT_ROOT not in sys.path:
    sys.path.insert(0, VGGT_ROOT)


class VGGTWrapper(nn.Module):
    """
    Args (construction):
        aggregator: an already-built Aggregator instance (test-injection path -- lets
            a standalone test use a tiny random-init Aggregator instead of
            downloading the real ~1B-parameter checkpoint). If None, loads the real
            pretrained VGGT-1B and keeps only its .aggregator, discarding the unused
            camera/depth/point/track heads to free their memory immediately (this
            wrapper never calls them).
        pretrained_name: HuggingFace Hub id for the real checkpoint, only used when
            aggregator is None.

    self.feat_dim: 2*embed_dim, inferred at construction time via a tiny dummy
        forward pass -- not hardcoded, so this stays correct for whatever
        checkpoint/config is actually loaded. Needed by callers to size
        DeformHeadMu/DeformHeadR's in_dim.
    """

    def __init__(self, aggregator=None, pretrained_name="facebook/VGGT-1B"):
        super().__init__()
        if aggregator is not None:
            self.aggregator = aggregator
        else:
            from vggt.models.vggt import VGGT
            full_model = VGGT.from_pretrained(pretrained_name)
            self.aggregator = full_model.aggregator
            # We only ever read .aggregator's dense tokens (Section 2's "pure dense
            # feature extractor" role) -- free the unused heads' memory immediately
            # rather than carrying them around unused for the rest of training.
            del full_model.camera_head, full_model.depth_head
            del full_model.point_head, full_model.track_head
            del full_model

        for p in self.aggregator.parameters():
            p.requires_grad_(False)
        self.aggregator.eval()

        self.patch_size = self.aggregator.patch_size

        # Infer feat_dim from a real (tiny) forward pass rather than hardcoding
        # 2*embed_dim -- self-verifying at runtime, matching this project's "confirm
        # against the real thing, don't assume" discipline, just done once at
        # construction instead of by reading source.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 3, self.patch_size, self.patch_size)
            dummy_out, _ = self.aggregator(dummy)
            last = [x for x in dummy_out if x is not None][-1]
            self.feat_dim = last.shape[-1]

    def forward(self, images_prev, images_curr):
        """
        Args:
            images_prev, images_curr: (1, N_cam, 3, H_pad14, W_pad14), raw [0,1] RGB,
                already padded to a multiple of self.patch_size -- i.e. exactly
                PadRawImagesForVGGT's output (metas["vggt_image_wh"]'s corresponding
                image tensor), NOT the existing ResNet-path imgs tensor.

        Returns:
            feat_prev, feat_curr: (1, N_cam, self.feat_dim, H', W') dense per-camera
                feature maps, H'=H_pad14/patch_size, W'=W_pad14/patch_size -- same
                (1, N_cam, C, H, W) convention DirectProjectionSampler's
                ms_img_feats/out_dpt_multiscale already use, ready for grid_sample.
        """
        assert images_prev.shape == images_curr.shape, (
            f"VGGTWrapper: images_prev {tuple(images_prev.shape)} and images_curr "
            f"{tuple(images_curr.shape)} must have identical shape -- both frames "
            f"padded via the same PadRawImagesForVGGT(size_divisor={self.patch_size})."
        )
        B, N_cam, _, H_pad, W_pad = images_prev.shape
        assert H_pad % self.patch_size == 0 and W_pad % self.patch_size == 0, (
            f"VGGTWrapper: input spatial dims ({H_pad},{W_pad}) not divisible by "
            f"patch_size={self.patch_size} -- these must be PadRawImagesForVGGT's "
            f"output, not the existing (32-padded) imgs tensor."
        )

        # Both frames fed together, one sequence -- Section 2's confirmed decision.
        images_seq = torch.cat([images_prev, images_curr], dim=1)  # (B, 2*N_cam, 3, H, W)

        with torch.no_grad():
            output_list, patch_start_idx = self.aggregator(images_seq)

        # Final cached layer only (Liam's confirmed decision -- see module docstring).
        last = [x for x in output_list if x is not None][-1]
        patch_tokens = last[:, :, patch_start_idx:, :]  # strip camera/register tokens

        H14 = H_pad // self.patch_size
        W14 = W_pad // self.patch_size
        patch_tokens = patch_tokens.reshape(B, 2 * N_cam, H14, W14, self.feat_dim)
        patch_tokens = patch_tokens.permute(0, 1, 4, 2, 3)  # (B, 2*N_cam, C, H', W')

        feat_prev = patch_tokens[:, :N_cam]
        feat_curr = patch_tokens[:, N_cam:]
        return feat_prev, feat_curr
