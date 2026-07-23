"""
scripts/train_stage_a.py

Root-cause fix for the Phase 5 exit-checklist failure: Stage B's do-nothing-vs-trained
comparison showed ~0% IoU on both branches, including static classes (driveable_surface,
manmade, sidewalk) that need no motion compensation at all. Root cause (confirmed,
EXPERIMENT_LOG.md): Stage A itself was NEVER actually trained -- every prior run
(Phase 2's single-frame overfit, Phase 5's wiring test, train_stage1.py) called
segmentor.init_weights() fresh, meaning only img_backbone had real pretrained weights
(r101_dcn_fcos3d_pretrain.pth); lifter/encoder/head -- everything that actually shapes
Gaussians into a real occupancy prediction -- started from random init every time.
Phase 3, which would have covered real Stage A training, was subsumed by Phase 2's
forward-pass check, not a training run. No .pth checkpoint beyond the pretrained
backbone was ever found on disk (confirmed via `find`).

This script trains Stage A for real, reusing GaussianFormer3D's own train.py mechanics
(optimizer/scheduler/loss construction, confirmed via source read) rather than
reimplementing them -- but building the dataloader via direct Occ4DGSDataset
instantiation (batch_size=1, no DataLoader/collate_fn) instead of train.py's
get_dataloader/cfg.train_dataset_config registry pattern, since Occ4DGSDataset was
always a direct-instantiation adapter (Phase 2 design), never wired into that registry.

Deliberate scope/deviations from train.py, each logged rather than silently changed:
  - SCENES=["scene-0061"] only for this first run (Decision 1: confirm convergence on
    1-2 scenes before scaling to all 10 -- same discipline as every prior phase).
  - No gradient accumulation (grad_accumulation=1) -- Phase 2 confirmed ample VRAM
    headroom (2.84GB peak) at batch_size=1, no need for it at this scale.
  - warmup_t scaled to ~5% of this run's actual total iterations, not train.py's
    literal warmup_iters=500 default. At SCENES=1 scene (~38 iters/epoch) x 24 epochs
    (~912 total iterations), a flat 500-iteration warmup would consume more than half
    the entire run just ramping to full LR -- clearly wrong at this dataset scale,
    where their default was presumably tuned for full nuScenes trainval (thousands of
    iterations/epoch). Logged as an explicit, reasoned deviation, not silent drift.
  - No MeanIoU validation loop in this script -- Decision 3 concluded the loss-curve
    convergence signal is the right check here; the real generalization check happens
    afterward, by re-running train_stage1.py's do-nothing-vs-trained comparison against
    this script's resulting checkpoint instead of a fresh init_weights() model.

Run from repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python scripts/train_stage_a.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_cwd = os.getcwd()
from run_stage_a_frame0 import build_pipeline, to_batch_of_one, GF3D_ROOT  # noqa: E402
os.chdir(_cwd)

import torch  # noqa: E402
from mmengine import Config  # noqa: E402
from mmengine.optim import build_optim_wrapper  # noqa: E402
from mmseg.models import build_segmentor  # noqa: E402
from timm.scheduler import CosineLRScheduler  # noqa: E402

sys.path.insert(0, GF3D_ROOT)
import model  # noqa: E402,F401
from loss import OPENOCC_LOSS  # noqa: E402

from src.datasets.nuscenes_mini import load_nuscenes  # noqa: E402
from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402

import json  # noqa: E402


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = ["scene-0061"]  # Decision 1: 1-2 scenes first, confirm convergence
GRAD_ACCUMULATION = 1     # no accumulation needed at this VRAM budget (Phase 2: 2.84GB peak)
WARMUP_FRACTION = 0.05    # scaled warmup, see module docstring
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "experiments", "stage_a_checkpoints")

# Override, not an edit to configs/occ4dgs_mini_occ3d_gs6400.py's own max_epochs=24 --
# that file is explicitly "mirrored from GaussianFormer3D/config/, kept in sync
# manually" (README.md), so editing it directly risks a future re-sync silently
# reverting this. First real Stage A training run (24 epochs) was still visibly
# improving at the final epoch (22.221 -> 22.203, not flattened) -- scaling to 60
# epochs (matching the project's other established epoch-count convention, e.g.
# stage_b_temporal.yaml's stage_1_warmup) to see if it actually converges rather than
# stopping while still improving.
N_EPOCHS_OVERRIDE = 60


def to_cuda(batch):
    out = {"imgs": batch["imgs"].cuda(), "points": [t.cuda() for t in batch["points"]]}
    out["metas"] = {k: v.cuda() for k, v in batch["metas"].items()}
    out["dpt"] = batch["dpt"].cuda() if batch["dpt"] is not None else None
    return out


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        full_frame_index = json.load(f)
    frame_index = {s: full_frame_index[s] for s in SCENES}

    dataset = Occ4DGSDataset(
        nusc, frame_index,
        os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
        os.path.join(REPO_ROOT, "data", "occ3d_gts"),
        pipeline=build_pipeline(),
    )
    print(f"{SCENES}: {len(dataset)} frames")

    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    max_epochs = N_EPOCHS_OVERRIDE  # see module-level constant's comment
    print(f"max_epochs: {max_epochs} (overriding config's max_epochs={cfg.max_epochs} "
          f"at the script level, not editing the shared config file)")
    my_model = build_segmentor(cfg.model)
    my_model.init_weights()  # loads cfg.load_from's pretrained img_backbone; everything
                              # else (lifter/encoder/head) starts random -- THIS is what
                              # this script's training actually fixes.
    my_model = my_model.cuda()
    my_model.train()
    n_params = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    print(f"Training {n_params:,} parameters (all of Stage A, nothing frozen)")

    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    total_iters = len(dataset) * max_epochs
    warmup_t = max(1, int(WARMUP_FRACTION * total_iters))
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_iters,
        lr_min=cfg.optimizer["optimizer"]["lr"] * 0.1,
        cycle_limit=1,
        warmup_t=warmup_t,
        warmup_lr_init=1e-6,
        t_in_epochs=False,
    )
    print(f"Total iterations: {total_iters}, warmup: {warmup_t} "
          f"({WARMUP_FRACTION*100:.0f}% -- scaled down from train.py's default "
          f"warmup_iters=500, see module docstring)")

    torch.cuda.reset_peak_memory_stats()
    global_iter = 0
    best_loss = float("inf")
    for epoch in range(max_epochs):
        epoch_losses = []
        optimizer.zero_grad()
        for idx in range(len(dataset)):
            sample = dataset[idx]
            batch = to_cuda(to_batch_of_one(sample))

            result_dict = my_model(
                imgs=batch["imgs"], metas=batch["metas"],
                points=batch["points"], dpt=batch["dpt"],
            )

            loss_input = {}
            for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                loss_input.update({loss_input_key: result_dict[loss_input_val]})
            loss, loss_dict = loss_func(loss_input)
            (loss / GRAD_ACCUMULATION).backward()

            global_iter += 1
            if global_iter % GRAD_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step_update(global_iter)

            epoch_losses.append(loss.item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  mean_loss={mean_loss:.5f}  lr={lr_now:.2e}  "
              f"peak_vram={peak_vram_gb:.2f}GB")

        torch.save(my_model.state_dict(),
                   os.path.join(CHECKPOINT_DIR, "stage_a_last.pth"))
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(my_model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "stage_a_best.pth"))
            print(f"  -> new best ({best_loss:.5f}), saved stage_a_best.pth")

    print(f"\nDone. Checkpoints in {CHECKPOINT_DIR}: stage_a_last.pth, stage_a_best.pth")
    print("Next: re-run train_stage1.py's do-nothing-vs-trained comparison, loading "
          "stage_a_best.pth into the segmentor instead of a fresh init_weights() model.")


if __name__ == "__main__":
    main()