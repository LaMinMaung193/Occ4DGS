"""
L_lidar: geometric consistency between warped Gaussian means mu_t and the observed LiDAR
point cloud P_t at frame t (nearest-Gaussian Chamfer-style term, or depth-consistency against
GaussianFormer3D's own LiDAR depth-map machinery -- pick whichever reuses more existing code).
See docs/design_doc_v2.md Section 4.

TODO(Phase 6): implement, then verify via ablate-to-zero (expect drift from LiDAR geometry)
"""
