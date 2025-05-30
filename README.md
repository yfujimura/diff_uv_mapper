# diff_uv_mapper
This code generates textured mesh from ply file containing colored vertexes and mesh, e.g. estimated by 2DGS.
![example](example/example.png)

## Requirements
```
# Pytorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast

# Other dependencies
pip install -r requirements.txt
```
## Usage
This code supports both NeRF and COLMAP datasets. At first, please estimate colored vertexes and mesh using [2DGS](https://github.com/hbb1/2d-gaussian-splatting) (e.g., fuse_post.ply). Then, a textured mesh is esitmated as follows:
```
# output textured mesh
python diff_uv_mapping.py -m data/chair/fuse_post.ply -s data/chair/

# optimize texture and mesh
python diff_uv_mapping.py -m data/chair/fuse_post.ply -s data/chair/ --run_optimization

# optimize texture and mesh on images rendered by 2DGS
python diff_uv_mapping.py -m data/chair/fuse_post.ply -s data/chair/ -r data/chair/renders --run_optimization

# optimize only texture
python diff_uv_mapping.py -m data/chair/fuse_post.ply -s data/chair/ --run_optimization --optimize_only_texture
```

## Acknowledgments
This code is based on the following great works:
- https://github.com/3DTopia/LGM/blob/main/convert.py
- https://github.com/ashawkey/kiuikit
- https://github.com/graphdeco-inria/gaussian-splatting
- https://github.com/hbb1/2d-gaussian-splatting
