"""
Stage A: GaussianFormer3D primitive Gaussian generation. Run ONCE per sequence, on frame 0.

See docs/design_doc_v2.md Section 1 for the full derivation:
    1.3 voxel-to-Gaussian initialization
    1.4 sparse convolution self-encoding
    1.5 LiDAR-guided 3D deformable attention (two-stage sampling, DFA3D-style)
    1.6 iterative refinement (4 blocks)

Camera backbone (ResNet50+FPN) and LiDAR voxel encoder defined here are REUSED by Stage B
(src/models/stage_b_temporal/) for current-frame encoding — do not duplicate them there.

TODO(Phase 2): implement per configs/stage_a_gaussianformer3d.yaml
"""
