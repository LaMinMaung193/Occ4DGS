"""
L_tv: penalizes CHANGE in predicted motion frame-to-frame (an acceleration/jitter penalty),
NOT the motion itself -- adapted from TED-4DGS's deformation-bank TV loss to this feedforward
(predicted, not free-parameter) setting. See docs/design_doc_v2.md Section 4.

    L_tv = mean_i [ || delta_mu_t^i - delta_mu_{t-1}^i ||_1 + || delta_r_t^i - delta_r_{t-1}^i ||_1 ]

TODO(Phase 6): implement, then verify via ablate-to-zero (expect visibly jittery trajectories)
"""
