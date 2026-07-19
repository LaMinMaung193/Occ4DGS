"""
Phase 2 step 3: visualize Gaussian output to check for known failure modes
(position collapse, Z-axis collapse, scale saturation) before trusting this
config across all 10 scenes. Reuses model/data loading from run_stage_a_frame0.py.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_stage_a_frame0 import GF3D_ROOT, REPO_ROOT, build_pipeline, to_batch_of_one

from mmengine import Config
from mmseg.models import build_segmentor
import model  # noqa: registers custom modules
from src.datasets.nuscenes_mini import load_nuscenes
from src.datasets.occ4dgs_dataset import Occ4DGSDataset

OUT_DIR = os.path.join(REPO_ROOT, "experiments", "phase2_gaussian_viz")


def run_one_scene(model_obj, dataset, scene_name):
    idx = next(i for i, s in enumerate(dataset.samples) if s[0] == scene_name)
    sample = dataset[idx]
    batch = to_batch_of_one(sample)
    for k in ["imgs", "points"]:
        batch[k] = [t.cuda() for t in batch[k]] if isinstance(batch[k], list) else batch[k].cuda()
    for mk in batch["metas"]:
        batch["metas"][mk] = batch["metas"][mk].cuda()
    if batch["dpt"] is not None:
        batch["dpt"] = batch["dpt"].cuda()
    with torch.no_grad():
        out = model_obj(**batch)
    return out


def analyze_gaussian(gaussian, scene_name, n_g_expected):
    means = gaussian.means.detach().cpu().numpy()[0]
    scales = gaussian.scales.detach().cpu().numpy()[0]
    opacities = gaussian.opacities.detach().cpu().numpy()[0]

    G = means.shape[0]
    tag = ("includes empty gaussian" if G == n_g_expected + 1
           else "matches exactly" if G == n_g_expected else "UNEXPECTED COUNT")
    print(f"\n=== {scene_name}: G={G} (expected N_g={n_g_expected}, {tag})")

    real = slice(0, n_g_expected) if G > n_g_expected else slice(0, G)
    m, s, o = means[real], scales[real], opacities[real]

    for i, axis in enumerate("xyz"):
        print(f"  means  {axis}: mean={m[:,i].mean():.3f} std={m[:,i].std():.3f} "
              f"min={m[:,i].min():.3f} max={m[:,i].max():.3f}")
    for i, axis in enumerate("xyz"):
        print(f"  scales {axis}: mean={s[:,i].mean():.3f} std={s[:,i].std():.3f} "
              f"min={s[:,i].min():.3f} max={s[:,i].max():.3f}")
    print(f"  opacity : mean={o.mean():.3f} std={o.std():.3f} min={o.min():.3f} max={o.max():.3f}")

    warnings = []
    if m[:,2].std() < 0.05 * max(m[:,0].std(), m[:,1].std(), 1e-6):
        warnings.append("Z-AXIS COLLAPSE suspected: z std is <5% of x/y std")
    if max(m[:,0].std(), m[:,1].std(), m[:,2].std()) < 1.0:
        warnings.append("POSITION COLLAPSE suspected: all axes have <1m std over an 80m range")
    near_min = (s <= 0.011).mean()
    near_max = (s >= 1.79).mean()
    if near_min > 0.3 or near_max > 0.3:
        warnings.append(f"SCALE SATURATION suspected: {near_min:.0%} near min bound, {near_max:.0%} near max bound")
    if warnings:
        print("  !! WARNINGS:")
        for w in warnings:
            print(f"     - {w}")
    else:
        print("  OK: no collapse/saturation signatures detected")
    return m, s, o


def plot_scene(m, s, o, scene_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(15, 4))

    ax1 = fig.add_subplot(131, projection="3d")
    color = o[:, 0] if o.shape[-1] > 0 else "steelblue"
    ax1.scatter(m[:,0], m[:,1], m[:,2], c=color, s=1, cmap="viridis")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)"); ax1.set_zlabel("z (m)")
    ax1.set_title(f"{scene_name}: positions (color=opacity)")

    ax2 = fig.add_subplot(132)
    for i, axis in enumerate("xyz"):
        ax2.hist(s[:,i], bins=50, alpha=0.5, label=f"scale_{axis}")
    ax2.legend(); ax2.set_title("Scale distribution")

    ax3 = fig.add_subplot(133)
    ax3.hist(o[:,0] if o.shape[-1] > 0 else [], bins=50)
    ax3.set_title("Opacity distribution")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{scene_name}_gaussians.png")
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  saved plot: {out_path}")


def main():
    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    model_obj = build_segmentor(cfg.model)
    model_obj.init_weights()
    model_obj = model_obj.cuda().eval()

    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        frame_index = json.load(f)
    dataset = Occ4DGSDataset(
        nusc=nusc, frame_index=frame_index,
        nuscenes_root=os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
        occ3d_gts_root=os.path.join(REPO_ROOT, "data", "occ3d_gts"),
        pipeline=build_pipeline(),
    )

    scene_names = sorted(set(s[0] for s in dataset.samples))
    print(f"Checking {len(scene_names)} scenes: {scene_names}")
    for scene_name in scene_names:
        out = run_one_scene(model_obj, dataset, scene_name)
        m, s, o = analyze_gaussian(out["gaussian"], scene_name, n_g_expected=cfg.model["lifter"]["num_anchor"])
        plot_scene(m, s, o, scene_name, OUT_DIR)

    print(f"\n{'='*60}\nAll {len(scene_names)} scenes checked. Review warnings above.")


if __name__ == "__main__":
    main()