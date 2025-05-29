import time
import numpy as np
import cv2
import torch
import xatlas

def uv_mapping(v, f):
    print(f"[xatlas] Start UV mapping ...")
    
    start = time.time()
    
    atlas = xatlas.Atlas()
    atlas.add_mesh(v, f)
    chart_options = xatlas.ChartOptions()
    atlas.generate(chart_options=chart_options)
    vmapping, ft, vt = atlas[0]

    end = time.time()
    print(f"[xatlas] Finish UV mapping: {end - start:.3f} sec.")
    
    return vmapping, ft, vt

def uv_padding(image, mask, padding = None, backend = 'knn'):
    """padding the uv-space texture image to avoid seam artifacts in mipmaps.

    Args:
        image (Union[Tensor, ndarray]): texture image, float, [H, W, C] in [0, 1].
        mask (Union[Tensor, ndarray]): valid uv region, bool, [H, W].
        padding (int, optional): padding size into the unmasked region. Defaults to 0.1 * max(H, W).
        backend (Literal[&#39;knn&#39;, &#39;cv2&#39;], optional): algorithm backend, knn is faster. Defaults to 'knn'.

    Returns:
        Union[Tensor, ndarray]: padded texture image. float, [H, W, C].
    """
    
    if torch.is_tensor(image):
        image_input = image.detach().cpu().numpy()
    else:
        image_input = image

    if torch.is_tensor(mask):
        mask_input = mask.detach().cpu().numpy()
    else:
        mask_input = mask
    
    if padding is None:
        H, W = image_input.shape[:2]
        padding = int(0.1 * max(H, W))
    
    # padding backend
    if backend == 'knn':

        from sklearn.neighbors import NearestNeighbors
        from scipy.ndimage import binary_dilation, binary_erosion

        inpaint_region = binary_dilation(mask_input, iterations=padding)
        inpaint_region[mask_input] = 0

        search_region = mask_input.copy()
        not_search_region = binary_erosion(search_region, iterations=2)
        search_region[not_search_region] = 0

        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
        _, indices = knn.kneighbors(inpaint_coords)

        inpaint_image = image_input.copy()
        inpaint_image[tuple(inpaint_coords.T)] = inpaint_image[tuple(search_coords[indices[:, 0]].T)]

    elif backend == 'cv2':
        # kind of slow
        inpaint_image = cv2.inpaint(
            (image_input * 255).astype(np.uint8),
            (~mask_input * 255).astype(np.uint8),
            padding,
            cv2.INPAINT_TELEA,
        ).astype(np.float32) / 255

    if torch.is_tensor(image):
        inpaint_image = torch.from_numpy(inpaint_image).to(image)
    
    return inpaint_image

def align_v_to_vt(v, f, vt, ft, vmapping=None):
    """ remap v/f and vn/fn to vt/ft.

    Args:
        vmapping (np.ndarray, optional): the mapping relationship from f to ft. Defaults to None.
    """
    if vmapping is None:
        ft = ft.view(-1).long()
        f = f.view(-1).long()
        vmapping = torch.zeros(vt.shape[0], dtype=torch.long, device=ft.device)
        vmapping[ft] = f # scatter, randomly choose one if index is not unique

    v = v[vmapping]
    f = ft
    
    return v, f, vt, ft