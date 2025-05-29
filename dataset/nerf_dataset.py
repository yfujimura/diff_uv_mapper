import os
import json
from PIL import Image
import numpy as np
import torch
from einops import rearrange

class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, white_background=True, renders_path=None):
        super().__init__()

        self.dataset_path = dataset_path
        self.bg = np.array([1.,1.,1.]) if white_background else np.array([0,0,0])
        
        with open(os.path.join(dataset_path, "transforms_train.json"), "r") as json_file:
            meta = json.load(json_file)

        self.frames = meta["frames"]
        self.fovx = self.fovy = 180 * meta["camera_angle_x"] / np.pi
        
    def __getitem__(self, index):
        frame = self.frames[index]

        rgba = np.array(Image.open(os.path.join(self.dataset_path, frame["file_path"] + ".png"))) / 255.
        image, alpha = rgba[:,:,:3], rgba[:,:,3:]
        image = alpha * image + (1 - alpha) * self.bg
        image = rearrange(torch.from_numpy(image).float(), "h w c -> c h w")

        c2w = torch.tensor(frame["transform_matrix"]) 
        
        return {"image": image, "fovx": self.fovx, "fovy": self.fovy, "c2w": c2w}

    def __len__(self):
        return len(self.frames) 