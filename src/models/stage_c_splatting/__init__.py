"""
Stage C: Gaussian-to-voxel splatting -> occupancy prediction. Identical to GaussianFormer3D's
own splatting module (Eq. 1-3 in that paper); operates the same whether fed Stage A's G_0
(Phase 3 smoke test) or Stage B's recursively-deformed G_t (Phase 5+).

TODO(Phase 3): implement per docs/design_doc_v2.md Section 3
"""
