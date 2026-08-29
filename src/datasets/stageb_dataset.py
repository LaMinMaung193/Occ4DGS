"""
src/datasets/stageb_dataset.py

Stage B training dataset: wraps GaussianFormer3D's own real NuScenesDataset
(loading real camera/LiDAR/depth for the "next moving keyframe") together with
each scene's cached G_0 and the real pose_prev/pose_curr needed for
transform_anchor_for_projection -- returning everything one single-step
training sample needs.

Built on scripts/build_stageb_manifest.py's output: a paired info pkl (frame 0
+ next moving keyframe, for 791 of 850 scenes) and a manifest recording which
index is which per scene.

Confirmed against real source (dataset/dataset.py) before writing this: the
underlying dataset SORTS self.keyframes by (scene_token, zero-padded index) --
not insertion order. Rather than assume this sorting keeps each scene's two
entries at a predictable relative position, we scan the sorted keyframes list
explicitly once at init and record each scene's (frame0_flat_idx,
next_frame_flat_idx) pair directly -- no dependence on assumed ordering.
"""
import json
import os

import torch
from torch.utils.data import Dataset

from dataset.utils import get_lidar2global


class StageBTrainingDataset(Dataset):
    def __init__(self, underlying_dataset, manifest_path, g0_cache_dir, raw_infos):
        """
        underlying_dataset: GaussianFormer3D's real NuScenesDataset, built from
            the paired info pkl (build_stageb_manifest.py's OUT_INFOS_PKL).
        manifest_path: build_stageb_manifest.py's OUT_MANIFEST (scene_token ->
            {next_idx, translation_m}).
        g0_cache_dir: directory of cached G_0 .pt files, one per scene.
        raw_infos: the same {scene_token: [frame_info, ...]} dict used to build
            the manifest -- needed for real pose lookup (calib/pose dicts).
        """
        self.underlying = underlying_dataset
        self.g0_cache_dir = g0_cache_dir
        self.raw_infos = raw_infos

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        scene_to_flat_indices = {}
        for flat_idx, (scene_token, frame_idx) in enumerate(self.underlying.keyframes):
            scene_to_flat_indices.setdefault(scene_token, []).append((frame_idx, flat_idx))

        self.samples = []
        n_mismatched = 0
        for scene_token, entries in scene_to_flat_indices.items():
            if scene_token not in self.manifest:
                continue
            if len(entries) != 2:
                n_mismatched += 1
                continue
            entries.sort(key=lambda x: x[0])
            (frame_idx_a, flat_a), (frame_idx_b, flat_b) = entries
            assert frame_idx_a == 0, f"expected frame 0 first, got {frame_idx_a} for {scene_token}"
            assert frame_idx_b == self.manifest[scene_token]["next_idx"], (
                f"manifest/dataset next_idx mismatch for {scene_token}: "
                f"{frame_idx_b} vs {self.manifest[scene_token]['next_idx']}"
            )
            self.samples.append((scene_token, flat_a, flat_b))

        if n_mismatched > 0:
            print(f"WARNING: {n_mismatched} scenes had != 2 entries in the "
                  f"underlying dataset, skipped")
        print(f"StageBTrainingDataset: {len(self.samples)} valid training samples")

    def __len__(self):
        return len(self.samples)

    def _get_real_pose(self, scene_token, frame_idx):
        info = self.raw_infos[scene_token][frame_idx]
        lidar_entry = info["data"]["LIDAR_TOP"]
        lidar2global = get_lidar2global(lidar_entry["calib"], lidar_entry["pose"])
        return torch.from_numpy(lidar2global).float()

    def __getitem__(self, idx):
        scene_token, flat_frame0_idx, flat_next_idx = self.samples[idx]

        data_next = self.underlying[flat_next_idx]

        pose_prev = self._get_real_pose(scene_token, 0)
        pose_curr = self._get_real_pose(scene_token, self.manifest[scene_token]["next_idx"])

        g0_data = torch.load(
            os.path.join(self.g0_cache_dir, f"{scene_token}.pt"), map_location="cpu"
        )

        return {
            "scene_token": scene_token,
            "data_next": data_next,
            "pose_prev": pose_prev,
            "pose_curr": pose_curr,
            "g0_means": g0_data["means"].squeeze(0),
            "g0_scales": g0_data["scales"].squeeze(0),
            "g0_rotations": g0_data["rotations"].squeeze(0),
            "g0_opacities": g0_data["opacities"].squeeze(0),
            "g0_semantics": g0_data["semantics"].squeeze(0),
            "translation_m": self.manifest[scene_token]["translation_m"],
        }
