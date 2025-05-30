import os
import sys
import numpy as np
from PIL import Image
import torch
from einops import rearrange
from typing import NamedTuple
import math

from utils.colmap_utils import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text, qvec2rotmat

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        #R = np.transpose(qvec2rotmat(extr.qvec))
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
    

class ColmapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, renders_path=None):
        super().__init__()

        try:
            cameras_extrinsic_file = os.path.join(dataset_path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(dataset_path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(dataset_path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(dataset_path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        cam_infos = readColmapCameras(cam_extrinsics, cam_intrinsics, os.path.join(dataset_path, "images"))
        self.cam_infos = sorted(cam_infos.copy(), key = lambda x : x.image_name)

        self.renders_path = renders_path

    def __getitem__(self, index):
        cam_info = self.cam_infos[index]

        if self.renders_path is None:
            rgba = np.array(Image.open(cam_info.image_path)) / 255.
            image, alpha = rgba[:,:,:3], rgba[:,:,3:]
            image = rearrange(torch.from_numpy(image).float(), "h w c -> c h w")
        else:
            image = np.array(Image.open(os.path.join(self.renders_path, f"{index:05}.png"))) / 255.
            image = rearrange(torch.from_numpy(image).float(), "h w c -> c h w")
            

        fovx = 180 * cam_info.FovX / np.pi
        fovy = 180 * cam_info.FovY / np.pi

        w2c = torch.eye(4)
        w2c[:3,:3] = torch.from_numpy(cam_info.R)
        w2c[:3,3] = torch.from_numpy(cam_info.T)
        
        c2w = w2c.inverse()
        # change from COLMAP (Y down, Z forward) to OpenGL/Blender camera axes (Y up, Z back)
        c2w[:3, 1:3] *= -1

        return {"image": image, "fovx": fovx, "fovy": fovy, "c2w": c2w}

    def __len__(self):
        return len(self.cam_infos) 