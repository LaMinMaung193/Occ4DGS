"""
scripts/plot_gaussians.py

Renders BEV comparisons of the raw GAUSSIAN PRIMITIVES (not occupancy
predictions) from render_gaussians.py's saved .npz files: do-nothing /
Dynamic 3DGS (L=2) / Static 3DGS, plus a 4th panel colored by per-Gaussian
displacement magnitude (do-nothing -> deformed) using a continuous colormap
-- shows WHERE the network is making its biggest changes, rather than a
cluttered arrow field.

Filters out Gaussians classified as "empty" (raw semantics argmax index 16,
per the reasoned mapping noted in render_gaussians.py -- not exhaustively
confirmed against source). If driveable_surface/vegetation don't appear in
geometrically plausible places, that's a concrete signal this mapping is off.

Pure CPU/numpy/matplotlib -- no GPU needed.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "/media/user/1TSSD/min/stageb_training/gaussian_data_L2"
OUT_DIR = "/media/user/1TSSD/min/stageb_training/gaussian_plots_L2"

CLASS_NAMES = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
               'manmade', 'vegetation']
EMPTY_RAW_INDEX = 16

CLASS_COLORS = {
    0: "#e6194b", 1: "#f58231", 2: "#ffe119", 3: "#bfef45", 4: "#3cb44b",
    5: "#42d4f4", 6: "#4363d8", 7: "#911eb4", 8: "#f032e6", 9: "#a9a9a9",
    10: "#808000", 11: "#469990", 12: "#9a6324", 13: "#800000", 14: "#000075",
    15: "#aaffc3",
}


def render_bev(ax, means, classes, title):
    mask = classes != EMPTY_RAW_INDEX
    means_f = means[mask]
    classes_f = classes[mask]

    if len(means_f) == 0:
        ax.set_title(f"{title}\n(no non-empty Gaussians)", fontsize=9)
        return

    order = np.argsort(means_f[:, 2])
    means_f = means_f[order]
    classes_f = classes_f[order]

    colors = [CLASS_COLORS.get(int(c), "#cccccc") for c in classes_f]
    ax.scatter(means_f[:, 0], means_f[:, 1], c=colors, s=2.0, marker="o", linewidths=0)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n({len(means_f):,} Gaussians)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def render_displacement(ax, means, displacement, title):
    order = np.argsort(displacement)
    means_o = means[order]
    disp_o = displacement[order]

    sc = ax.scatter(means_o[:, 0], means_o[:, 1], c=disp_o, s=2.0, marker="o",
                     linewidths=0, cmap="inferno", vmin=0, vmax=np.percentile(displacement, 95))
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n(mean {displacement.mean():.2f}m, max {displacement.max():.2f}m)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"Found {len(files)} scene files")

    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        scene_token = str(data["scene_token"])

        fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
        render_bev(axes[0], data["donothing_means"], data["donothing_class"], "Do-Nothing 3DGS")
        render_bev(axes[1], data["deformed_means"], data["deformed_class"], "Dynamic 3DGS (L=2)")
        render_bev(axes[2], data["static_means"], data["static_class"], "Static 3DGS")
        sc = render_displacement(axes[3], data["donothing_means"], data["displacement"],
                                  "Displacement: Do-Nothing -> Dynamic")

        cbar = fig.colorbar(sc, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.set_label("Displacement (m)", fontsize=8)

        class_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=CLASS_COLORS[i], markersize=7,
                                     label=CLASS_NAMES[i]) for i in range(16)]
        fig.legend(handles=class_handles, loc="lower center", ncol=8, fontsize=7.5,
                   bbox_to_anchor=(0.5, -0.08), title="Semantic classes (raw Gaussian argmax)",
                   title_fontsize=8)

        fig.suptitle(f"Scene {scene_token[:16]}... — Raw Gaussian Comparison", fontsize=13, y=1.08)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, os.path.basename(fpath).replace(".npz", ".png"))
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Rendered -> {out_path}")

    print(f"\nAll {len(files)} scenes rendered to {OUT_DIR}")


if __name__ == "__main__":
    main()
