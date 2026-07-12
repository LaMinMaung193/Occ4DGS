"""
Stage 1 (frozen warm-up) training entrypoint. Stage A frozen; trains only
motion_hypernet + phi_mu + phi_r. See configs/stage_b_temporal.yaml: stage_1_warmup and
docs/IMPLEMENTATION_ROADMAP.md Phase 5.

TODO(Phase 5): implement training loop, AMP + gradient checkpointing per README.md decision table
"""
