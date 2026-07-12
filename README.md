# Occ4DGS: Dynamic 4D Gaussian Splatting for Occupancy Prediction in Autonomous Driving

Feedforward temporal deformation of GaussianFormer3D primitives for dynamic 3D semantic
occupancy prediction, on nuScenes v1.0-mini (10 scenes) with Occ3D-nuScenes GT.

CCU Autonomous Driving Perception Lab, advised by Prof. Rachael (Jui-Chiu) Chiang.

See `docs/IMPLEMENTATION_ROADMAP.md` for the full phase-by-phase plan and exit checklists,
`EXPERIMENT_LOG.md` for the running research log, and `docs/design_doc_v2.md` +
`docs/dataset_compute_addendum.md` for the architecture and data-source rationale.

## Assigned defaults (decided, not pending professor approval)

These were previously open decisions; they are now fixed as working defaults and adjusted
only via pilot runs, not left unresolved:

| Decision | Value | Rationale (short) |
|---|---|---|
| `N_g` (num. Gaussians) | **6,400** | Fits single RTX 3090 24GB with Stage B unroll; matches GaussianFormer-2's own ablation showing modest IoU cost vs 25,600 at ~4x less memory. Revisit upward (12,800) only after Phase 8 if VRAM allows. |
| Camera backbone | **ResNet50 + FPN** | Matches existing QG-Fusion code (reuse), much lighter than paper's ResNet101-DCN. |
| Image resolution | **450×800** (half of paper's 900×1600) | Memory budget; revisit if quality suffers. |
| Stage 1 (frozen warm-up) LR | **1e-4** (AdamW, cosine schedule) for HyperNet + Φ_μ + Φ_r | Matches GaussianFormer3D's own nuScenes LR for new modules. |
| Stage 2 (joint fine-tune) LR | **1e-5** for Stage A (GaussianFormer3D) params, **5e-5** for temporal module | 10x lower LR for the already-converged generator; ratio, not absolute value, is what matters. |
| Unroll window (Stage 1) | **2 frames** | Conservative starting point for VRAM; confirmed via Phase 5 profiling. |
| Unroll window (Stage 2) | **3 frames** | Increased once Stage 1 stability confirmed. |
| Epochs | Stage 1: **60**, Stage 2: **40** | Small dataset (10 scenes, ~400 frames) — epochs are cheap; adjust based on validation curve, not fixed in stone. |
| Batch size | **1 sequence/step**, gradient accumulation ×4 | Effective batch 4 on single GPU. |
| Precision | **AMP (fp16)** | Near-mandatory on 24GB. |
| Frozen vs. joint | **Staged**: Stage 1 frozen Stage A → Stage 2 joint fine-tune | Already argued in `design_doc_v2.md` §5; kept as main config, frozen-only kept as ablation. |
| GT source | **Occ3D-nuScenes** | Existing verified loader from 3DGS project; see `dataset_compute_addendum.md`. |
| Scene set | **nuScenes v1.0-mini (10 scenes)** | Self-contained, matches professor's scope exactly. |

All of the above are recorded here so any future run's config diverging from this table is a
deliberate, logged decision (see `EXPERIMENT_LOG.md`), not an accidental default drift.
