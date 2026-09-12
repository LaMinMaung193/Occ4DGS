"""
scripts/plot_qualitative.py (v2)

Renders BEV comparisons: GT / do-nothing baseline / Dynamic 3DGS (ours) /
Static 3DGS, plus a 5th difference-map panel showing exactly where Dynamic
3DGS improves on or regresses from the do-nothing baseline, relative to GT --
directly answers "where does it help" rather than relying on the eye to spot
subtle differences between two similar-looking panels.

Terminology (v2): "Stage B" -> "Dynamic 3DGS (ours)" (parallels "Static
3DGS" cleanly); "oracle" dropped entirely (unexplained jargon for a general
reader) -- both changed per direct request after reviewing v1's output.

Pure CPU/numpy/matplotlib -- no GPU needed, safe to run anytime.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "/media/user/1TSSD/min/stageb_training/qualitative_data_L2_pretrained_release"
OUT_DIR = "/media/user/1TSSD/min/stageb_training/qualitative_plots_L2_pretrained_release"

CLASS_NAMES = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
               'manmade', 'vegetation']
EMPTY_LABEL = 17

CLASS_COLORS = {
    1: "#e6194b", 2: "#f58231", 3: "#ffe119", 4: "#bfef45", 5: "#3cb44b",
    6: "#42d4f4", 7: "#4363d8", 8: "#911eb4", 9: "#f032e6", 10: "#a9a9a9",
    11: "#808000", 12: "#469990", 13: "#9a6324", 14: "#800000", 15: "#000075",
    16: "#aaffc3",
}

DIFF_COLORS = {
    "improved": "#2ca02c",
    "regressed": "#d62728",
    "agree_correct": "#cfe8cf",
    "agree_wrong": "#e0e0e0",
}


def render_bev(ax, xyz, labels, title):
    mask = labels != EMPTY_LABEL
    xyz_f = xyz[mask]
    labels_f = labels[mask]

    if len(xyz_f) == 0:
        ax.set_title(f"{title}\n(no occupied voxels)", fontsize=9)
        return

    order = np.argsort(xyz_f[:, 2])
    xyz_f = xyz_f[order]
    labels_f = labels_f[order]

    colors = [CLASS_COLORS.get(int(l), "#cccccc") for l in labels_f]
    ax.scatter(xyz_f[:, 0], xyz_f[:, 1], c=colors, s=1.1, marker="s", linewidths=0)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n({len(xyz_f):,} occupied voxels)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def render_diff(ax, xyz, gt, pred_donothing, pred_dynamic, title):
    mask = gt != EMPTY_LABEL
    xyz_f = xyz[mask]
    gt_f = gt[mask]
    donothing_f = pred_donothing[mask]
    dynamic_f = pred_dynamic[mask]

    donothing_correct = donothing_f == gt_f
    dynamic_correct = dynamic_f == gt_f

    colors = np.empty(len(xyz_f), dtype=object)
    colors[dynamic_correct & ~donothing_correct] = DIFF_COLORS["improved"]
    colors[~dynamic_correct & donothing_correct] = DIFF_COLORS["regressed"]
    colors[dynamic_correct & donothing_correct] = DIFF_COLORS["agree_correct"]
    colors[~dynamic_correct & ~donothing_correct] = DIFF_COLORS["agree_wrong"]

    order = np.zeros(len(xyz_f), dtype=int)
    order[colors == DIFF_COLORS["agree_wrong"]] = 0
    order[colors == DIFF_COLORS["agree_correct"]] = 1
    order[colors == DIFF_COLORS["regressed"]] = 2
    order[colors == DIFF_COLORS["improved"]] = 3
    sort_idx = np.argsort(order)

    ax.scatter(xyz_f[sort_idx, 0], xyz_f[sort_idx, 1], c=list(colors[sort_idx]),
               s=1.4, marker="s", linewidths=0)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    n_improved = int((dynamic_correct & ~donothing_correct).sum())
    n_regressed = int((~dynamic_correct & donothing_correct).sum())
    ax.set_title(f"{title}\n(+{n_improved:,} improved / -{n_regressed:,} regressed)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"Found {len(files)} scene files")

    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        scene_token = str(data["scene_token"])
        xyz = data["sampled_xyz"]
        gt = data["sampled_label"]
        pred_donothing = data["pred_donothing"]
        pred_dynamic = data["pred_stageb"]
        pred_static = data["pred_oracle"]

        fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
        render_bev(axes[0], xyz, gt, "Ground Truth")
        render_bev(axes[1], xyz, pred_donothing, "Do-nothing baseline")
        render_bev(axes[2], xyz, pred_dynamic, "Dynamic 3DGS (ours)")
        render_bev(axes[3], xyz, pred_static, "Static 3DGS")
        render_diff(axes[4], xyz, gt, pred_donothing, pred_dynamic,
                    "Dynamic 3DGS vs. do-nothing")

        class_handles = [plt.Line2D([0], [0], marker="s", color="w",
                                     markerfacecolor=CLASS_COLORS[i + 1], markersize=7,
                                     label=CLASS_NAMES[i]) for i in range(16)]
        diff_handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=DIFF_COLORS["improved"],
                       markersize=7, label="Improved"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=DIFF_COLORS["regressed"],
                       markersize=7, label="Regressed"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=DIFF_COLORS["agree_correct"],
                       markersize=7, label="Both correct"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=DIFF_COLORS["agree_wrong"],
                       markersize=7, label="Both wrong"),
        ]

        fig.legend(handles=class_handles, loc="lower left", ncol=4, fontsize=7.5,
                   bbox_to_anchor=(0.02, -0.08), title="Semantic classes", title_fontsize=8)
        fig.legend(handles=diff_handles, loc="lower right", ncol=1, fontsize=7.5,
                   bbox_to_anchor=(0.99, -0.08), title="Diff map", title_fontsize=8)

        fig.suptitle(f"Scene {scene_token[:16]}... — BEV Occupancy Comparison", fontsize=13, y=1.08)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, os.path.basename(fpath).replace(".npz", ".png"))
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Rendered -> {out_path}")

    print(f"\nAll {len(files)} scenes rendered to {OUT_DIR}")


if __name__ == "__main__":
    main()
