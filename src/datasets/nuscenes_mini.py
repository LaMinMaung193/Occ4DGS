"""
Loads the 10 nuScenes v1.0-mini scenes via nuscenes-devkit and returns ordered per-scene
frame lists (sample tokens + sensor file paths).

See docs/IMPLEMENTATION_ROADMAP.md Phase 1, step 1.

TODO(Phase 1):
    - load NuScenes(version='v1.0-mini', dataroot=data/nuscenes_mini)
    - for each of the 10 scenes, walk sample -> next chain to build ordered frame list
    - return: List[scene_name -> List[sample_token]]
"""
