"""
Motion hypernetwork: predicts a compact multi-resolution motion grid M_t from the CURRENT
frame's fused camera+LiDAR feature volume F^3D_t.

Unlike original 4DGC (M_t as free per-scene parameters, optimized via test-time gradient
descent), here M_t is the OUTPUT of this network, so the same weights generalize across
frames and scenes without any per-scene fitting. This is the core novel module of the paper
— see docs/design_doc_v2.md Section 2.3.

TODO(Phase 4/5): implement per configs/stage_b_temporal.yaml: motion_hypernet
"""
