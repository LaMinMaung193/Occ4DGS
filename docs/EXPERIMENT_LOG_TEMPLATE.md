# Experiment Log Template

Copy this block into `EXPERIMENT_LOG.md` for every run — passed, failed, or partial. The
point is that `EXPERIMENT_LOG.md` becomes the single source of truth for "what did we
actually try and what happened," so paper writing (Phase 10) and any professor check-in
pulls directly from here rather than from memory.

```markdown
## [Phase N] Run ID: YYYYMMDD-HHMM-short-description

- **Git commit:** <hash>
- **Config file(s):** configs/...
- **Command:**
  ```
  <exact command run>
  ```
- **Hardware:** RTX 3090 24GB, <peak VRAM observed>
- **Hypothesis / what this run tests:**
  <one or two sentences>
- **Results:**

  | Metric | Value |
  |---|---|
  | IoU (overall) | |
  | mIoU (overall) | |
  | Per-class mIoU | (link to full table if long) |
  | Temporal flicker | |
  | FPS / latency | |
  | Peak VRAM | |

- **Observations:**
  <what you actually saw — plots, qualitative results, anything surprising>
- **Bugs / issues encountered & fixes:**
  <e.g. "position collapse at epoch 12, fixed by X" — mirrors the QG-Fusion debugging log style>
- **Decision / next step:**
  <what this run changes about the plan — proceed, retry with X changed, escalate to professor, etc.>
```

## Notes on using this well

- Log **failed** runs too — a run that diverges or shows no learning signal is exactly the
  kind of evidence the Limitations section and future ablation choices need.
- When a run deviates from the assigned defaults in the root `README.md` table, say so
  explicitly and why — don't let config drift go unrecorded.
- Keep one running summary table at the top of `EXPERIMENT_LOG.md` (see the seeded file) so
  you don't have to scroll through every entry to compare runs at a glance.
