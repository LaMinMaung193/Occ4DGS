"""
scripts/diagnose_vggt_param_coverage.py

Checks whether real training actually reached and updated every intended trainable
parameter in VGGTDeformableController -- motivated by a real, worth-checking finding:
train_stage1.py's optimizer is built as

    trainable_params = list(pool.parameters()) + list(hypernet.parameters())
                        + list(deform_mu.parameters()) + list(deform_r.parameters())

For the VGGT-deformable branch, hypernet/deform_mu/deform_r are nn.Identity()
placeholders (0 parameters each), so this reduces to list(pool.parameters()) --
but `pool` (VGGTDeformableController) also contains VGGTWrapper's ~1B frozen
parameters as registered submodules, since .parameters() does not filter by
requires_grad. AdamW's step() implementation skips any parameter whose .grad is
None (which frozen params always have, since requires_grad=False excludes them
from the backward graph), so this is expected to be harmless -- inefficient
(constructing a ~1B-entry Python list unnecessarily) but not incorrect. This script
verifies that claim directly on the real trained checkpoint, rather than trusting
the reasoning alone, and -- more importantly -- confirms every REAL trainable
parameter (all 4 blocks + initial_embed) actually changed from its random init,
which is the real question: did training actually reach everything it was supposed
to.

Run: PYTHONNOUSERSITE=1 python scripts/diagnose_vggt_param_coverage.py
(requires USE_VGGT_DEFORMABLE=True, and a saved checkpoint from a completed n=3 run)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.training.stage_b_engine import (  # noqa: E402
    REPO_ROOT, build_temporal_module, USE_VGGT_DEFORMABLE,
)


def main():
    assert USE_VGGT_DEFORMABLE, "Set USE_VGGT_DEFORMABLE=True in stage_b_engine.py first."

    ckpt_path = os.path.join(
        REPO_ROOT, "experiments", "stage_b_temporal_checkpoints", "stage1_warmup_temporal_n3_final.pth"
    )
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    print("Building a FRESH (untrained, freshly random-initialized) pool for comparison...")
    fresh_pool, _, _, _, _, _, _ = build_temporal_module()
    fresh_state = {k: v.clone().cpu() for k, v in fresh_pool.state_dict().items()}
    trained_state = ckpt["pool"]

    print(f"\nFresh state dict has {len(fresh_state)} keys, trained has {len(trained_state)} keys")
    assert set(fresh_state.keys()) == set(trained_state.keys()), (
        "KEY MISMATCH between fresh and trained state_dicts -- architecture drift "
        "between when the checkpoint was saved and now, or a real bug"
    )

    frozen_unchanged, frozen_changed = [], []
    trainable_unchanged, trainable_changed = [], []

    for name in fresh_state:
        is_frozen = name.startswith("vggt.aggregator.")
        changed = not torch.equal(fresh_state[name], trained_state[name].cpu())
        if is_frozen:
            (frozen_changed if changed else frozen_unchanged).append(name)
        else:
            (trainable_changed if changed else trainable_unchanged).append(name)

    print(f"\n=== FROZEN (VGGT aggregator) params ===")
    print(f"  Unchanged (expected): {len(frozen_unchanged)}")
    print(f"  Changed (SHOULD BE ZERO -- real bug if not): {len(frozen_changed)}")
    if frozen_changed:
        print(f"  First few unexpectedly-changed frozen params: {frozen_changed[:5]}")

    print(f"\n=== TRAINABLE (initial_embed + 4 blocks) params ===")
    print(f"  Changed (expected -- these received real gradient updates): {len(trainable_changed)}")
    print(f"  UNCHANGED (SUSPICIOUS -- did these receive gradient at all?): {len(trainable_unchanged)}")
    if trainable_unchanged:
        print(f"\n  Full list of unchanged trainable params (investigate these):")
        for name in trainable_unchanged:
            print(f"    {name}")

    total_trainable = len(trainable_changed) + len(trainable_unchanged)
    pct_changed = 100.0 * len(trainable_changed) / total_trainable if total_trainable else 0.0
    print(f"\n{pct_changed:.1f}% of trainable parameters changed after training "
          f"({len(trainable_changed)}/{total_trainable})")

    if frozen_changed:
        print("\n[FAIL] Some VGGT (frozen) parameters changed -- freezing is broken, a real bug.")
    else:
        print("\n[PASS] All VGGT (frozen) parameters are byte-identical to fresh init -- "
              "freezing works correctly, confirms the frozen-params-in-optimizer "
              "situation is harmless as reasoned (AdamW correctly skips them).")

    if trainable_unchanged:
        print("[FLAG] Some trainable parameters never changed -- worth investigating "
              "whether these are genuinely dead (e.g. the LAST block's query_update_mlp, "
              "already known/expected to be permanently unused -- see EXPERIMENT_LOG.md) "
              "or a real gradient-flow bug for anything else in this list.")
    else:
        print("[PASS] Every trainable parameter changed from its fresh init -- "
              "training reached all of them, no dead/unreached parameters found.")


if __name__ == "__main__":
    main()
