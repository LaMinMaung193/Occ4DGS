# Git Workflow

## Branching
- `main` — always the last known-good phase exit state. Only fast-forward-merge into `main`
  once a phase's exit checklist (see `docs/IMPLEMENTATION_ROADMAP.md`) is fully checked.
- One working branch per phase: `phase0-env-setup`, `phase1-data-index`,
  `phase2-stageA`, `phase3-stageC-smoketest`, `phase4-stageB-skeleton`,
  `phase5-stage1-warmup`, `phase6-losses`, `phase7-stage2-joint`, `phase8-evaluation`,
  `phase9-ablations`, `phase10-writing`.
- Merge each phase branch into `main` at phase exit, then tag (see below), then branch the
  next phase off the updated `main`.

## Commit message convention
```
<type>(<scope>): <short summary>

<optional longer body>
```
Types: `feat` (new capability), `fix` (bug fix), `exp` (experiment run / config sweep),
`docs` (documentation only), `refactor`, `chore` (housekeeping, deps, gitignore, etc.).
Scope: the module/phase touched, e.g. `stageA`, `stageB`, `dataset`, `losses`, `roadmap`.

Examples:
```
feat(stageB): add motion hypernet and per-Gaussian grid query
fix(dataset): correctly tag has_gt for mini_train index gap 0-38
exp(phase5): stage1 warmup run 003, window=2, 10 scenes, mIoU 18.4
docs(roadmap): update phase7 exit checklist after VRAM profiling
```

## Tags
Tag every phase exit exactly as named in `docs/IMPLEMENTATION_ROADMAP.md` (e.g.
`v0.0-phase0-env-verified`, `v0.5-phase5-stage1-trained`, ... `v1.0-submission`). Tags are
what you point the professor to when showing progress — "here's the tagged state phase 5
exited in" is a much stronger status update than a verbal summary.

```bash
git tag -a v0.0-phase0-env-verified -m "Phase 0 exit: env verified, data indexed"
git push origin v0.0-phase0-env-verified
```

## Initial setup on your machine (RTX 3090, this repo)

```bash
cd Occ4DGS
git init
git add -A
git commit -m "chore: initial project scaffold (Occ4DGS)"

# Create the remote on GitHub first (via web UI or `gh repo create`), then:
git remote add origin git@github.com:<your-username>/Occ4DGS.git
git branch -M main
git push -u origin main
```

If using GitHub CLI instead of the web UI:
```bash
gh repo create Occ4DGS --private --source=. --remote=origin --push
```

## Day-to-day loop
```bash
git checkout -b phaseN-short-name
# ... do the work, log runs in EXPERIMENT_LOG.md as you go, not all at the end ...
git add -A
git commit -m "feat(...): ..."
git push -u origin phaseN-short-name
# once phase exit checklist is fully checked:
git checkout main
git merge phaseN-short-name
git tag -a vX.Y-phaseN-... -m "..."
git push origin main --tags
```
