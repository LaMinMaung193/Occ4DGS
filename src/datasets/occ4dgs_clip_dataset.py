"""
src/datasets/occ4dgs_clip_dataset.py

Phase 5: wraps an existing Occ4DGSDataset (single, independently-indexed frames) into
ordered temporal clips of `unroll_window` consecutive frames, for Stage B's recursive
training (design_doc_v2.md Section 2, roadmap Phase 5 step 4).

Reuse, not reimplement: this class duplicates none of Occ4DGSDataset's loading logic
(pipeline, sensor paths, depth/occ GT resolution) -- it only groups existing
(scene_name, sample_token, occ_path_rel) entries into windows and delegates every
actual __getitem__ call back to the wrapped dataset.

Confirmed (EXPERIMENT_LOG.md, Phase 5 bridge investigation):
  - src/datasets/nuscenes_mini.py's build_all_scene_frames() walks first_sample_token ->
    next, so per-scene token lists are genuinely temporally ordered.
  - scripts/build_frame_index.py preserves that order verbatim when filtering by has_gt
    (list comprehension, no re-sort) -- so Occ4DGSDataset.samples is temporally ordered
    WITHIN each scene, PROVIDED consecutive kept entries are also temporally adjacent in
    the underlying nuScenes sample chain. True today because Phase 1 confirmed 100%
    has_gt coverage across all 10 scenes -- if that ever changes, a has_gt=False frame
    would open a gap that list-adjacency alone would not catch.

Because of that caveat, this class does NOT trust list-adjacency alone: it verifies
temporal adjacency explicitly via nuScenes' own sample["next"] pointer before forming a
window, so a future non-100%-coverage scene fails loudly (assertion) rather than silently
handing Stage B a "2-frame unroll" that is actually two temporally-unrelated frames.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.datasets.occ4dgs_dataset import Occ4DGSDataset  # noqa: E402


class Occ4DGSClipDataset:
    """
    Wraps an Occ4DGSDataset instance. Each item is a list of `unroll_window` dicts
    (whatever the wrapped dataset's __getitem__ already returns), in temporal order,
    guaranteed to be consecutive real nuScenes frames within one scene.

    Does NOT subclass torch.utils.data.Dataset directly to avoid implying it's a drop-in
    replacement for Occ4DGSDataset in existing single-frame code paths -- it is a
    different item shape (list of dicts, not one dict). Wrap with a thin torch Dataset
    adapter at the training-script level if/when a DataLoader is wired in (Phase 5 step 3
    onward), since collation of a list-of-dicts needs an explicit choice there anyway.
    """

    def __init__(self, base_dataset: Occ4DGSDataset, unroll_window: int):
        assert unroll_window >= 2, "a clip of 1 frame is not a temporal unroll"
        self.base = base_dataset
        self.unroll_window = unroll_window
        self.clips = self._build_clips()

    def _build_clips(self):
        # Group base_dataset.samples (flat list of (scene_name, sample_token, occ_path_rel))
        # by scene, preserving order, then verify + window.
        by_scene = {}
        for idx, (scene_name, sample_token, _occ_path_rel) in enumerate(self.base.samples):
            by_scene.setdefault(scene_name, []).append((idx, sample_token))

        clips = []
        for scene_name, entries in by_scene.items():
            for start in range(len(entries) - self.unroll_window + 1):
                window = entries[start:start + self.unroll_window]
                self._assert_temporally_adjacent(scene_name, window)
                clips.append([idx for idx, _tok in window])
        return clips

    def _assert_temporally_adjacent(self, scene_name, window):
        nusc = self.base.nusc
        for (_idx_a, tok_a), (_idx_b, tok_b) in zip(window[:-1], window[1:]):
            next_tok = nusc.get("sample", tok_a)["next"]
            assert next_tok == tok_b, (
                f"Occ4DGSClipDataset: {scene_name} samples {tok_a} -> {tok_b} are adjacent "
                f"in the has_gt-filtered list but NOT adjacent in nuScenes' own sample "
                f"chain (a has_gt=False gap exists between them). Refusing to silently "
                f"build a clip that would hand Stage B two non-consecutive frames as if "
                f"they were dt=1 apart. Re-run scripts/build_frame_index.py and check for "
                f"new has_gt=False entries before retrying."
            )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, clip_idx):
        indices = self.clips[clip_idx]
        return [self.base[i] for i in indices]