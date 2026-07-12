"""
Phi_mu and Phi_r: two lightweight shared MLPs mapping a per-Gaussian temporal feature z_t^i
(from trilinear-interpolating the predicted motion grid at each Gaussian's position, with
positional encoding per 4DGC Eq. 3) to a position delta and a rotation delta.

No Phi_s (scale) head, no compensated-Gaussian branch — confirmed dropped, see
docs/design_doc_v2.md Section 2.5.

Update rule (design_doc_v2.md Section 2.6):
    mu_t   = mu_{t-1} + delta_mu_t
    r_t    = normalize(delta_r_t (quat) composed with r_{t-1})   # composition, not addition
    s_t, alpha_t, c_t unchanged (time-invariant)

TODO(Phase 4/5): implement per configs/stage_b_temporal.yaml: deform_heads
"""
