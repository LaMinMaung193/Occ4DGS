"""
Phase 2 step 5 (flagged "important" in the roadmap): overfit Stage A on a single
frame to confirm the training path (loss, backward, optimizer) works end-to-end,
and get a reference mIoU number -- NOT a real training run, just a correctness
and sanity-baseline check before Phase 3.

Loss call convention confirmed from GaussianFormer3D/train.py lines 200-260:
    loss_input = {'metas': data}
    for k, v in cfg.loss_input_convertion.items(): loss_input[k] = result_dict[v]
    loss, loss_dict = loss_func(loss_input)
"""
import os, sys, json
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GF3D_ROOT = os.path.expanduser("~/Documents/min/GaussianFormer3D")
sys.path.insert(0, GF3D_ROOT)
sys.path.insert(0, REPO_ROOT)
os.chdir(GF3D_ROOT)

from mmengine import Config
from mmseg.models import build_segmentor
import model  # noqa: registers custom modules
from loss import OPENOCC_LOSS

from src.datasets.nuscenes_mini import load_nuscenes
from src.datasets.occ4dgs_dataset import Occ4DGSDataset
from run_stage_a_frame0 import build_pipeline, to_batch_of_one

def _first_shape(x):
    if isinstance(x, (list, tuple)):
        return f"list[{len(x)}] of {x[0].shape if hasattr(x[0], 'shape') else type(x[0])}"
    return x.shape


def main():
    cfg = Config.fromfile(os.path.join(GF3D_ROOT, "config", "occ4dgs_mini_occ3d_gs6400.py"))
    model_obj = build_segmentor(cfg.model)
    model_obj.init_weights()
    model_obj = model_obj.cuda()

    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    optimizer = torch.optim.AdamW(model_obj.parameters(),
                                   lr=cfg.optimizer["optimizer"]["lr"],
                                   weight_decay=cfg.optimizer["optimizer"]["weight_decay"])

    nusc = load_nuscenes(os.path.join(REPO_ROOT, "data", "nuscenes_mini"))
    with open(os.path.join(REPO_ROOT, "experiments", "phase1_frame_index.json")) as f:
        frame_index = json.load(f)
    dataset = Occ4DGSDataset(
        nusc=nusc, frame_index=frame_index,
        nuscenes_root=os.path.join(REPO_ROOT, "data", "nuscenes_mini"),
        occ3d_gts_root=os.path.join(REPO_ROOT, "data", "occ3d_gts"),
        pipeline=build_pipeline(),
    )
    idx = next(i for i, s in enumerate(dataset.samples) if s[0] == "scene-0061")
    sample = dataset[idx]
    batch = to_batch_of_one(sample)
    for k in ["imgs", "points"]:
        batch[k] = [t.cuda() for t in batch[k]] if isinstance(batch[k], list) else batch[k].cuda()
    for mk in batch["metas"]:
        batch["metas"][mk] = batch["metas"][mk].cuda()
    if batch["dpt"] is not None:
        batch["dpt"] = batch["dpt"].cuda()

    model_obj.train()
    n_iters = 200
    for it in range(n_iters):
        imgs_in = batch["imgs"].clone()
        dpt_in = batch["dpt"].clone() if batch["dpt"] is not None else None
        result_dict = model_obj(imgs=imgs_in, points=batch["points"],
                                 dpt=dpt_in, metas=batch["metas"])
        if it == 0:
            print("pred_occ:", _first_shape(result_dict["pred_occ"]))
            print("sampled_label:", _first_shape(result_dict["sampled_label"]))
            print("occ_mask:", _first_shape(result_dict["occ_mask"]) if result_dict["occ_mask"] is not None else None)

        loss_input = {"metas": batch["metas"]}
        for k, v in cfg.loss_input_convertion.items():
            loss_input[k] = result_dict[v]
        loss, loss_dict = loss_func(loss_input)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_obj.parameters(), cfg.grad_max_norm)
        optimizer.step()

        if it % 20 == 0 or it == n_iters - 1:
            print(f"iter {it:3d}  loss={loss.item():.4f}  " +
                  "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items() if hasattr(v, "item")))

    # quick reference mIoU on the same (overfit) frame
    model_obj.eval()
    with torch.no_grad():
        imgs_in = batch["imgs"].clone()
        dpt_in = batch["dpt"].clone() if batch["dpt"] is not None else None
        result_dict = model_obj(imgs=imgs_in, points=batch["points"],
                                 dpt=dpt_in, metas=batch["metas"])
    pred_occ_final = result_dict["pred_occ"][-1] if isinstance(result_dict["pred_occ"], (list, tuple)) else result_dict["pred_occ"]
    label_final = result_dict["sampled_label"][-1] if isinstance(result_dict["sampled_label"], (list, tuple)) else result_dict["sampled_label"]
    pred = pred_occ_final.argmax(dim=1)  # class dimension is dim=1: [1, 18, 640000] -> [1, 640000]
    label = label_final
    ious = []
    for c in range(1, 17):  # skip class 0? -- verify against config's manual_class_weight indexing; classes 1-16 are the "real" semantic classes per empty_label=17 convention
        pred_c = (pred == c)
        label_c = (label == c)
        inter = (pred_c & label_c).sum().item()
        union = (pred_c | label_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    miou = sum(ious) / len(ious) if ious else 0.0
    print(f"\nReference single-frame overfit mIoU (classes 1-16): {miou:.4f} (n_classes_present={len(ious)})")


if __name__ == "__main__":
    main()