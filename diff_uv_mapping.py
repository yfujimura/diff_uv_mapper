# This code generates textured mesh from ply file containing colored vertexes and mesh, e.g. estimated by 2DGS.
# Usage:
#  - python diff_uv_mapping.py -m data/chair/fuse_post.ply -s data/chair/ # output textured mesh
#  - python diff_uv_mapping.py -m data/chair/fuse_post.ply -s data/chair/ --run_optimization # optimize texture and mesh
#  - python diff_uv_mapping.py -m data/chair/fuse_post.ply -s data/chair/ -r data/chair/renders --run_optimization # optimize texture and mesh on images rendered by 2DGS
#
# This code is based on the following great works:
#   - https://github.com/3DTopia/LGM/blob/main/convert.py
#   - https://github.com/ashawkey/kiuikit
#   - https://github.com/graphdeco-inria/gaussian-splatting
#   - https://github.com/hbb1/2d-gaussian-splatting

import os
from argparse import ArgumentParser
import tqdm
import random
import numpy as np
import cv2
import open3d as o3d
import torch
from torch import nn
import torch.nn.functional as F
import nvdiffrast.torch as dr
import pygltflib

from utils.uv_utils import uv_mapping, uv_padding, align_v_to_vt
from utils.optim_utils import inverse_sigmoid, OptimConfig
from utils.camera_utils import get_perspective
from dataset import get_dataset

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--mesh_path", type=str, required=True, help="ply file containing colored vertexes and mesh")
    parser.add_argument("-s", "--scene_path", type=str, required=True, help="Path to NeRF or COLMAP scene")
    parser.add_argument("-r", "--renders_path", type=str, default=None, help="Path to rendered images. If this is not None, the optimization is done on not the original images but the rendered images")
    parser.add_argument("-o", "--output_path", type=str, default="output", help="Output directory")
    parser.add_argument("--decimate_ratio", type=float, default=0.3, help="Decimation ratio of faces")
    parser.add_argument("--run_optimization", action='store_true', help="If True, texture and vertexes are optimized")
    parser.add_argument("--train_iters", type=int, default=1000, help="Number of training iters")
    parser.add_argument("--weight", type=float, default=1e+4, help="Weight for regularizing vertex offsets")
    args = parser.parse_args()
    return args

class DiffUVMapper():

    def __init__(
        self, 
        mesh_path,
        scene_path,
        renders_path=None,
        decimate_ratio=0.3,
        texture_resolution=1024,
        padding=20,
    ):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if decimate_ratio > 0:
            target_triangle_count = int(len(mesh.triangles) * decimate_ratio)
            mesh = mesh.simplify_quadric_decimation(target_triangle_count)

        self.v = np.asarray(mesh.vertices)
        self.c = np.asarray(mesh.vertex_colors)
        self.f = np.asarray(mesh.triangles)

        self.vmapping, self.ft, self.vt = uv_mapping(self.v, self.f)

        self.v = torch.from_numpy(self.v.astype(np.float32)).cuda()
        self.c = torch.from_numpy(self.c.astype(np.float32)).cuda()
        self.f = torch.from_numpy(self.f.astype(np.int32)).cuda()
        self.ft = torch.from_numpy(self.ft.astype(np.int32)).cuda()
        self.vt = torch.from_numpy(self.vt.astype(np.float32)).cuda()
        self.vmapping = self.vmapping.astype(np.int32)

        self.glctx = dr.RasterizeCudaContext()

        h = w = texture_resolution
        uv = self.vt * 2.0 - 1.0 # uvs to range [-1, 1]
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]
        rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), self.ft, (h, w)) # [1, h, w, 4]
        mask, _ = dr.interpolate(torch.ones_like(self.v[:, :1]).unsqueeze(0), rast, self.f) # [1, h, w, 1]
        albedo, _ = dr.interpolate(self.c.unsqueeze(0), rast, self.f)  # [1, h, w, 3]

        mask = mask.squeeze()
        albedo = albedo.squeeze()
        
        mask = mask > 0
        albedo = uv_padding(albedo, mask, padding)

        self.albedo = nn.Parameter(inverse_sigmoid(albedo)).cuda()
        self.offset = nn.Parameter(torch.zeros_like(self.v))

        self.train_dataset = get_dataset(scene_path, renders_path=renders_path)

    def render_mesh(self, height, width, fovy, fovx, pose):
        h, w = height, width
        proj = torch.from_numpy(get_perspective(fovy, aspect=fovx/fovy)).float().cuda()

        v = self.v + self.offset
    
        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T
    
        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.f, (h, w))
    
        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous() # [1, H, W, 1]
        alpha = dr.antialias(alpha, rast, v_clip, self.f).clamp(0, 1).squeeze(-1).squeeze(0) # [H, W] important to enable gradients!
    
        texc, texc_db = dr.interpolate(self.vt.unsqueeze(0), rast, self.ft, rast_db=rast_db, diff_attrs='all')
        image = torch.sigmoid(dr.texture(self.albedo.unsqueeze(0), texc, uv_da=texc_db)) # [1, H, W, 3]
    
        image = image.view(1, h, w, 3)
        # image = dr.antialias(image, rast, v_clip, f).clamp(0, 1)
        image = image.squeeze(0).permute(2, 0, 1).contiguous() # [3, H, W]
        image = alpha * image + (1 - alpha)
        return image

    def optimize(
        self, 
        optim_config,
    ):
        train_iters = optim_config.train_iters
        lr = optim_config.lr
        weight = optim_config.weight
        
        optimizer = torch.optim.Adam([
            {'params': self.albedo, 'lr': lr["albedo"]},
            {'params': self.offset, 'lr': lr["offset"]},
        ])
        
        idxs = list(range(len(self.train_dataset)))
        pbar = tqdm.trange(train_iters)
        for i in pbar:
            if len(idxs) == 0:
                idxs = list(range(len(self.train_dataset)))
            idx = idxs.pop(random.randint(0, len(idxs) - 1))

            data = self.train_dataset[idx]
            image_gt = data["image"].cuda()
            fovx = data["fovx"]
            fovy = data["fovy"]
            pose = data["c2w"].cuda()
            
            image_pred = self.render_mesh(image_gt.shape[1], image_gt.shape[2], fovy, fovx, pose)
            loss_mse = F.mse_loss(image_pred, image_gt)
            loss_offset = (self.offset**2).mean()

            loss = loss_mse + weight * loss_offset
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"MSE = {loss_mse.item():.6f}")

        self.v = self.v + self.offset

    def write_glb(self, path):
        v, f, vt, ft = align_v_to_vt(self.v, self.f, self.vt, self.ft, self.vmapping)
    
        f_np = f.detach().cpu().numpy().astype(np.uint32)
        f_np_blob = f_np.flatten().tobytes()
    
        v_np = v.detach().cpu().numpy().astype(np.float32)
        v_np_blob = v_np.tobytes()
    
        blob = f_np_blob + v_np_blob
        byteOffset = len(blob)
    
        # base mesh
        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[pygltflib.Mesh(primitives=[pygltflib.Primitive(
                # indices to accessors (0 is triangles)
                attributes=pygltflib.Attributes(
                    POSITION=1,
                ),
                indices=0,
            )])],
            buffers=[
                pygltflib.Buffer(byteLength=len(f_np_blob) + len(v_np_blob))
            ],
            # buffer view (based on dtype)
            bufferViews=[
                # triangles; as flatten (element) array
                pygltflib.BufferView(
                    buffer=0,
                    byteLength=len(f_np_blob),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER, # GL_ELEMENT_ARRAY_BUFFER (34963)
                ),
                # positions; as vec3 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob),
                    byteLength=len(v_np_blob),
                    byteStride=12, # vec3
                    target=pygltflib.ARRAY_BUFFER, # GL_ARRAY_BUFFER (34962)
                ),
            ],
            accessors=[
                # 0 = triangles
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.UNSIGNED_INT, # GL_UNSIGNED_INT (5125)
                    count=f_np.size,
                    type=pygltflib.SCALAR,
                    max=[int(f_np.max())],
                    min=[int(f_np.min())],
                ),
                # 1 = positions
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT, # GL_FLOAT (5126)
                    count=len(v_np),
                    type=pygltflib.VEC3,
                    max=v_np.max(axis=0).tolist(),
                    min=v_np.min(axis=0).tolist(),
                ),
            ],
        )
    
        # append texture info
        if vt is not None:
    
            vt_np = vt.detach().cpu().numpy().astype(np.float32)
            vt_np_blob = vt_np.tobytes()
    
            albedo = torch.sigmoid(self.albedo).detach().cpu().numpy()
            albedo = (albedo * 255).astype(np.uint8)
            albedo = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)
            albedo_blob = cv2.imencode('.png', albedo)[1].tobytes()
    
            # update primitive
            gltf.meshes[0].primitives[0].attributes.TEXCOORD_0 = 2
            gltf.meshes[0].primitives[0].material = 0
    
            # update materials
            gltf.materials.append(pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorTexture=pygltflib.TextureInfo(index=0, texCoord=0),
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                ),
                alphaMode=pygltflib.OPAQUE,
                alphaCutoff=None,
                doubleSided=True,
            ))
    
            gltf.textures.append(pygltflib.Texture(sampler=0, source=0))
            gltf.samplers.append(pygltflib.Sampler(magFilter=pygltflib.LINEAR, minFilter=pygltflib.LINEAR_MIPMAP_LINEAR, wrapS=pygltflib.REPEAT, wrapT=pygltflib.REPEAT))
            gltf.images.append(pygltflib.Image(bufferView=3, mimeType="image/png"))
    
            # update buffers
            gltf.bufferViews.append(
                # index = 2, texcoords; as vec2 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=byteOffset,
                    byteLength=len(vt_np_blob),
                    byteStride=8, # vec2
                    target=pygltflib.ARRAY_BUFFER,
                )
            )
    
            gltf.accessors.append(
                # 2 = texcoords
                pygltflib.Accessor(
                    bufferView=2,
                    componentType=pygltflib.FLOAT,
                    count=len(vt_np),
                    type=pygltflib.VEC2,
                    max=vt_np.max(axis=0).tolist(),
                    min=vt_np.min(axis=0).tolist(),
                )
            )
    
            blob += vt_np_blob 
            byteOffset += len(vt_np_blob)
    
            gltf.bufferViews.append(
                # index = 3, albedo texture; as none target
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=byteOffset,
                    byteLength=len(albedo_blob),
                )
            )
    
            blob += albedo_blob
            byteOffset += len(albedo_blob)
    
            gltf.buffers[0].byteLength = byteOffset
    
            
        # set actual data
        gltf.set_binary_blob(blob)
    
        # glb = b"".join(gltf.save_to_bytes())
        gltf.save(path)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    diff_uv_mapper = DiffUVMapper(args.mesh_path, args.scene_path, renders_path=args.renders_path, decimate_ratio=args.decimate_ratio)
    if args.run_optimization:
        optim_config = OptimConfig(
            args.train_iters,
            {
                "albedo": 1e-2,
                "offset": 1e-3,
            },
            args.weight,
        )
        diff_uv_mapper.optimize(optim_config)
        diff_uv_mapper.write_glb(os.path.join(args.output_path, f"mesh_w_optim_{args.train_iters}.glb"))
        print(f"Saved to", os.path.join(args.output_path, f"mesh_w_optim_{args.train_iters}.glb"))
    else:
        diff_uv_mapper.write_glb(os.path.join(args.output_path, "mesh_wo_optim.glb"))
        print(f"Saved to", os.path.join(args.output_path, "mesh_wo_optim.glb"))

