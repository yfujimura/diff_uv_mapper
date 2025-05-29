import os
from .colmap_dataset import ColmapDataset
from .nerf_dataset import NeRFDataset

def get_dataset(scene_path, renders_path=None):
    if os.path.exists(os.path.join(scene_path, "sparse")):
        return  ColmapDataset(scene_path, renders_path=renders_path)
    elif os.path.exists(os.path.join(scene_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        return  NeRFDataset(scene_path, renders_path=renders_path)
    else:
        assert False, "Could not recognize scene type!"