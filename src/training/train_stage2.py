"""
Stage 2 (joint fine-tune) training entrypoint. Unfreezes Stage A at a 10x lower LR than the
temporal module; initializes from Stage 1's best checkpoint. See
configs/stage_b_temporal.yaml: stage_2_joint and docs/IMPLEMENTATION_ROADMAP.md Phase 7.

TODO(Phase 7): implement, watch for early-epoch instability (see roadmap Phase 7 step 4)
"""
