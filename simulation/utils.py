import torch
from pathlib import Path
import numpy as np
from torch.nn import functional as F
import trimesh
from scipy.spatial.transform import Rotation
from PIL import Image
from typing import List, Optional, Tuple
from torchvision.io import write_video
from torchvision.transforms.functional import pil_to_tensor
import cv2
import matplotlib.pyplot as plt
import io
from torchvision.transforms.functional import gaussian_blur
from scipy import ndimage
from skimage.morphology import remove_small_objects
import os
import glob
from glob import glob
import imageio.v2 as imageio
from typing import List, Tuple, Optional

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    PointLights,
)
from pytorch3d.transforms import Transform3d
from torchvision.utils import save_image

PRESET_Z_VALUE = 0.0 # 10.0

def load_simulation_state(load_path):
    """
    Load simulation state from a .pth file.

    Args:
        load_path (str or Path): Path to the .pth file to load

    Returns:
        dict: A dictionary containing the loaded simulation state, including config, obj_gaussians, env_gaussians, delta_time, and save_dir.
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"File not found: {load_path}")

    state = torch.load(load_path, map_location='cpu')
    print(f"Simulation state loaded from {load_path}")

    # Convert config back to OmegaConf if needed
    if isinstance(state['config'], dict):
        try:
            from omegaconf import OmegaConf
            state['config'] = OmegaConf.create(state['config'])
        except ImportError:
            print("OmegaConf is not available. Config will remain as a dictionary.")

    return state

def pt3d_to_gs(xyz, no_z_offset=False):
    z_offset = 0.0 if no_z_offset else PRESET_Z_VALUE
    # xyz: [..., 3]
    # transform xyz coordinates to genesis coordinates
    # pt3d: x - left, y - up, z - forward
    # genesis: x - right, y - forward, z - up
    if isinstance(xyz, torch.Tensor):
        xyz_new = xyz.clone()
        xyz_new[..., 0] = -1 * xyz[..., 0]
        xyz_new[..., 1] = xyz[..., 2]
        xyz_new[..., 2] = xyz[..., 1]
    elif isinstance(xyz, np.ndarray):
        xyz_new = xyz.copy()
        xyz_new[..., 0] = -1 * xyz[..., 0]
        xyz_new[..., 1] = xyz[..., 2]
        xyz_new[..., 2] = xyz[..., 1]
    else:
        raise ValueError(f"Input type {type(xyz)} is not supported")

    xyz_new[..., 2] += z_offset
    return xyz_new

def gs_to_pt3d(xyz, no_z_offset=False):
    z_offset = 0.0 if no_z_offset else PRESET_Z_VALUE
    # xyz: [..., 3]
    # transform xyz coordinates to pt3d coordinates
    # pt3d: x - left, y - up, z - forward
    # genesis: x - right, y - forward, z - up
    if isinstance(xyz, torch.Tensor):
        xyz_new = xyz.clone()
        xyz_new[..., 0] = -1 * xyz[..., 0]
        xyz_new[..., 1] = xyz[..., 2]
        xyz_new[..., 2] = xyz[..., 1]
    elif isinstance(xyz, np.ndarray):
        xyz_new = xyz.copy()
        xyz_new[..., 0] = -1 * xyz[..., 0]
        xyz_new[..., 1] = xyz[..., 2]
        xyz_new[..., 2] = xyz[..., 1]
    else:
        raise ValueError(f"Input type {type(xyz)} is not supported")

    xyz_new[..., 1] -= z_offset
    return xyz_new

def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws

def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics

def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics

def FOV_to_intrinsics(fov, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics


def export_trimesh_from_vertices_faces(vertices, faces, output_path, vertex_colors=None, process=False):
    """
    Export trimesh from vertices and faces
    
    Args:
        vertices: numpy array or torch tensor of shape [N, 3]
        faces: numpy array or torch tensor of shape [F, 3] 
        output_path: str, path to save the mesh (e.g., 'mesh.obj', 'mesh.ply')
        vertex_colors: optional numpy array or torch tensor of shape [N, 3] with RGB values [0-1]
        process: bool, whether to process the mesh (clean, merge vertices, etc.)
    """
    
    # Convert to numpy if needed
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()
    if vertex_colors is not None and torch.is_tensor(vertex_colors):
        vertex_colors = vertex_colors.detach().cpu().numpy()
    
    # Create trimesh object
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=process  # Set to False to avoid automatic processing
    )
    
    # Add vertex colors if provided
    if vertex_colors is not None:
        # Ensure colors are in [0, 255] range for trimesh
        if vertex_colors.max() <= 1.0:
            vertex_colors = (vertex_colors * 255).astype(np.uint8)
        else:
            vertex_colors = vertex_colors.astype(np.uint8)
        
        # Set vertex colors
        mesh.visual.vertex_colors = vertex_colors
    
    # Export the mesh
    mesh.export(output_path)
    print(f"Mesh exported to: {output_path}")
    
    return mesh


def pose_to_transform_matrix(pos, quat):
    """
    Convert position and quaternion to 4x4 transformation matrix for Genesis
    
    Args:
        pos: [x, y, z] position vector 
        quat: [w, x, y, z] quaternion (Genesis format)
    
    Returns:
        4x4 transformation matrix (numpy array)
    """
    # Convert to numpy arrays (handle torch tensors if needed)
    if hasattr(pos, 'cpu'):  # torch tensor
        pos = pos.cpu().numpy()
    if hasattr(quat, 'cpu'):  # torch tensor
        quat = quat.cpu().numpy()
    
    pos = np.array(pos, dtype=np.float64)
    quat = np.array(quat, dtype=np.float64)
    
    # Genesis uses [w, x, y, z] format, convert to scipy [x, y, z, w] format
    quat_scipy = quat[[1, 2, 3, 0]]  # [x, y, z, w]
    
    # Normalize quaternion
    quat_scipy = quat_scipy / np.linalg.norm(quat_scipy)
    
    # Create rotation matrix from quaternion
    rot = Rotation.from_quat(quat_scipy)
    rot_matrix = rot.as_matrix()
    
    # Build 4x4 transformation matrix
    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = rot_matrix  # rotation part
    transform_matrix[:3, 3] = pos          # translation part
    
    return transform_matrix


def save_video_from_pil(
    frames: List[Image.Image],
    out_path: str,
    fps: int = 16,
    size: Optional[Tuple[int, int]] = None,   # (width, height). If None, use first frame size
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    yuv420p: bool = True,
) -> None:
    """
    Save a video from a list of PIL Images using torchvision.io.write_video.

    Args:
        frames: list of PIL Images (can be RGB/L/LA/RGBA; will be converted to RGB).
        out_path: output video path, e.g., 'out.mp4'.
        fps: frames per second.
        size: (W, H). If None, inferred from frames[0].
        codec: e.g., 'libx264', 'libx265', 'h264_nvenc' (if available), etc.
        crf: quality for x264/x265 (lower = better quality, larger file).
        preset: encoder speed/efficiency tradeoff ('ultrafast' ... 'placebo').
        yuv420p: if True, force yuv420p for broad player compatibility.
    """
    if not frames:
        raise ValueError("frames is empty")

    # Target size (width, height)
    if size is None:
        size = (frames[0].width, frames[0].height)
    W, H = size

    tensor_frames = []
    for im in frames:
        # Normalize to RGB and size
        im = im.convert("RGB")
        if (im.width, im.height) != (W, H):
            im = im.resize((W, H), Image.BICUBIC)

        # PIL -> torch uint8 tensor, shape (C,H,W) -> (H,W,C)
        t = pil_to_tensor(im).permute(1, 2, 0).contiguous()  # uint8 on CPU
        tensor_frames.append(t)

    # Stack to (T,H,W,C) uint8
    video = torch.stack(tensor_frames, dim=0)  # CPU uint8

    # Encoder options
    options = {"crf": str(crf), "preset": preset}
    if yuv420p:
        options["pix_fmt"] = "yuv420p"

    write_video(
        filename=out_path,
        video_array=video,      # (T,H,W,C), uint8 RGB
        fps=fps,
        video_codec=codec,
        options=options,
    )

def save_gif_from_image_folder(input_folder, gif_path, duration=0.1):
    image_exts = ('*.png', '*.jpg', '*.jpeg')
    input_images = []
    for ext in image_exts:
        input_images.extend(glob(os.path.join(input_folder, ext)))
    
    input_images = sorted(input_images)

    if not input_images:
        print("No images found in input folder.")
        return

    frames = []
    for img_path in input_images:
        try:
            img = imageio.imread(img_path)
            frames.append(img)
        except Exception as e:
            print(f"[ERROR] Skipping {img_path}: {e}")

    if frames:
        imageio.mimsave(gif_path, frames, duration=duration, loop=0)
        print(f"GIF saved to {gif_path}")
    else:
        print("No valid images to save as GIF.")


def resize_and_crop_pil(
    image: Image.Image,
    start_y=None,
    target_size: Tuple[int, int] = (832, 480),
    preserve_aspect: bool = False,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Resize image to target size.

    - Legacy mode (preserve_aspect=False): resize to a square and then crop vertically.
    - Letterbox mode (preserve_aspect=True): keep aspect ratio and pad to target size.
    """
    target_w, target_h = target_size

    if preserve_aspect:
        src_w, src_h = image.size
        scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        resized = image.convert("RGB").resize((new_w, new_h), resample=Image.BILINEAR)
        canvas = Image.new("RGB", (target_w, target_h), color=pad_color)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas

    # Legacy crop behavior used by existing pipeline.
    square_size = max(target_w, target_h)
    resized_image = image.resize((square_size, square_size), resample=Image.BILINEAR)

    start_x = max((square_size - target_w) // 2, 0)
    if start_y is None:
        start_y = max((square_size - target_h) // 2, 0)
    start_y = int(np.clip(start_y, 0, square_size - target_h))

    cropped_image = resized_image.crop((start_x, start_y, start_x + target_w, start_y + target_h))
    return cropped_image
    
def dilate_binary_mask(mask: np.ndarray, size=(512, 512), kernel_size=5, iterations=1):
    """
    Args:
        mask: np.ndarray, binary mask with values 0 or 1
        size: output image size (W, H)
        kernel_size: dilation kernel size (odd int)
        iterations: number of dilation iterations
    Returns:
        PIL.Image: dilated binary mask as RGB image with values 0 or 255
    """
    # Ensure mask is binary uint8 (0 or 255)
    mask = mask.detach().cpu().numpy()
    mask_uint8 = (mask.astype(np.uint8) * 255)

    # Resize to target size
    mask_resized = cv2.resize(mask_uint8, size, interpolation=cv2.INTER_NEAREST)

    # Apply dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_resized, kernel, iterations=iterations)

    # Convert to RGB and ensure it's still binary (0 or 255)
    dilated_binary = np.where(dilated > 0, 255, 0).astype(np.uint8)

    return dilated_binary

def smooth_segmentation_mask_255(
    mask: np.ndarray,
    blur_kernel_size: int = 15,
    blur_sigma: float = 5.0,
    threshold: int = 60,
    binary_output: bool = True,
    morph_close: bool = True,
    morph_kernel_size: int = 7,
    return_pil: bool = True
):
    """
    Smooth a 0-255 binary mask (uint8), soften edges to make blob-like shape.

    Args:
        mask (np.ndarray): Input mask with values 0 or 255, shape (H, W)
        blur_kernel_size (int): Size of Gaussian blur kernel
        blur_sigma (float): Sigma value for Gaussian blur
        threshold (int): Threshold value for binary output (0-255)
        binary_output (bool): If True, return 0/255 mask; if False, return soft mask
        morph_close (bool): Apply morphological closing to remove holes
        morph_kernel_size (int): Kernel size for morphological ops
        return_pil (bool): If True, return PIL image, else NumPy array

    Returns:
        PIL.Image or np.ndarray: Smoothed mask image (L mode)
    """
    assert mask.ndim == 2, "Mask must be 2D"
    assert np.issubdtype(mask.dtype, np.uint8), "Mask must be uint8"
    assert set(np.unique(mask)).issubset({0, 255}), "Mask must contain only 0 or 255"

    # Step 1: normalize to [0, 1]
    mask_norm = (mask / 255.0).astype(np.float32)

    # Step 2: blur
    blurred = cv2.GaussianBlur(mask_norm, (blur_kernel_size, blur_kernel_size), sigmaX=blur_sigma)

    # Step 3: threshold or scale
    if binary_output:
        mask_out = (blurred > (threshold / 255.0)).astype(np.uint8) * 255
    else:
        mask_out = np.clip(blurred * 255.0, 0, 255).astype(np.uint8)

    # Step 4: morphological closing (optional)
    if morph_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(mask_out).convert("L") if return_pil else mask_out

def save_depth_map(depth_map, file_name, vmin=None, vmax=None, save_clean=False):
    depth_map = np.squeeze(depth_map)
    if depth_map.ndim != 2:
        raise ValueError("Depth map after squeezing must be 2D.")

    dpi = 100  # Adjust this value if necessary
    figsize = (depth_map.shape[1] / dpi, depth_map.shape[0] / dpi)  # Width, Height in inches

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cax = ax.imshow(depth_map, cmap='viridis', vmin=vmin, vmax=vmax)

    if not save_clean:
        # Standard save with labels and color bar
        cbar = fig.colorbar(cax)
        ax.set_title("Depth Map")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
    else:
        # Clean save without labels, color bar, or axis
        plt.axis('off')
        ax.set_aspect('equal', adjustable='box')

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert('RGB')  # Convert to RGB
    img = img.resize((depth_map.shape[1], depth_map.shape[0]), Image.Resampling.LANCZOS)  # Resize to original dimensions
    img.save(file_name, format='png')
    buf.close()
    plt.close()

def soft_stitching(source_img, target_img, masks, blur_size=3, sigma=2.5):
    """
    Perform soft stitching between source and target images using multiple object masks.
    
    Args:
        source_img: Source image tensor
        target_img: Target image tensor
        masks: List of mask tensors or single mask tensor
        blur_size: Size of the Gaussian kernel, must be odd
        sigma: Standard deviation of the Gaussian kernel
    
    Returns:
        Stitched image tensor
    """
    
    # Handle single mask case - convert to list for uniform processing
    if not isinstance(masks, list):
        masks = [masks]
    
    # Combine all masks using element-wise maximum (union of all masks)
    # This ensures that any pixel covered by any mask is included
    combined_mask = masks[0].float()
    for mask in masks[1:]:
        combined_mask = torch.maximum(combined_mask, mask.float())
    
    # Adding padding to reduce edge effects during blurring
    padding = blur_size // 2
    soft_mask = F.pad(combined_mask, (padding, padding, padding, padding), mode='reflect')

    # Apply the Gaussian blur
    blurred_mask = gaussian_blur(soft_mask, kernel_size=(blur_size, blur_size), sigma=(sigma, sigma))

    # Remove the padding
    blurred_mask = blurred_mask[:, :, padding:-padding, padding:-padding]

    # Ensure the mask is within 0 and 1 after blurring
    blurred_mask = torch.clamp(blurred_mask, 0, 1)

    # Blend the images based on the blurred mask
    stitched_img = source_img * blurred_mask + target_img * (1 - blurred_mask)

    return stitched_img

def save_mask_kps(mask, kps_h, kps_w, save_path):

    mask_np = mask.squeeze().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(mask_np, cmap='gray', interpolation='nearest')

    plt.scatter(kps_w.cpu().numpy(), kps_h.cpu().numpy(), c='red', s=10, label='Points')
    plt.axis('off')
    plt.legend()

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.clf()
    plt.close()

def extract_foreground_depth_torch(
    depth: torch.Tensor,
    mask: torch.Tensor,
    r: int = 10,
    background: str = "keep",
    clip_percentiles=(1.0, 99.0),
    max_iters: int | None = None,

    # new robust parameters
    use_bilateral: bool = True,        # whether to use "approximate bilateral" weighted (based on the range weight of the neighborhood median)
    sigma_s: float = 1.0,              # spatial term
    sigma_r_scale: float = 3.0,        # range term scale = sigma_r_scale * (local MAD+eps)
    min_count: int = 3,                # minimum number of known neighbors needed to fill
    mad_guard: float = 6.0,            # if the neighborhood MAD is too large (> mad_guard times the global MAD), this round will not be filled
    expand_clip_k: float = 3.0,        # expand clip k
):
    assert depth.ndim == 2 and mask.ndim == 2, "depth and mask must be 2D"
    H, W = depth.shape
    device = depth.device
    depth = depth.to(torch.float32)
    mask  = mask.to(torch.bool)

    if r > 0:
        k = 2 * r + 1
        w = torch.ones((1, 1, k, k), device=device, dtype=torch.float32)
        m = mask.float().unsqueeze(0).unsqueeze(0)
        conv = F.conv2d(m, w, padding=r)
        core = (conv == k * k).squeeze(0).squeeze(0)
    else:
        core = mask.clone()

    if core.sum() == 0:
        core = mask.clone()

    depth_fg = torch.full_like(depth, float("nan"))
    core_vals = depth[core]
    if core_vals.numel() > 0:
        if clip_percentiles is not None:
            lo = torch.quantile(core_vals, clip_percentiles[0] / 100.0)
            hi = torch.quantile(core_vals, clip_percentiles[1] / 100.0)
            core_vals = core_vals.clamp(lo, hi)
        depth_fg[core] = core_vals

    eps = 1e-6
    if core_vals.numel() > 0:
        core_med = core_vals.median()
        core_mad = (core_vals - core_med).abs().median() + eps
        if clip_percentiles is not None:
            lo_clip = lo - expand_clip_k * core_mad
            hi_clip = hi + expand_clip_k * core_mad
        else:
            lo_clip = core_med - expand_clip_k * core_mad
            hi_clip = core_med + expand_clip_k * core_mad
    else:
        lo_clip, hi_clip = -float('inf'), float('inf')

    if max_iters is None:
        max_iters = H + W

    k3 = torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32)
    known = torch.isfinite(depth_fg)

    def neighborhood_stack(x_bool_or_float):
        x = x_bool_or_float.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        patches = F.unfold(x, kernel_size=3, padding=1)  # (1, 9, H*W)
        patches = patches.view(9, H, W).permute(1, 2, 0)  # (H,W,9)
        return patches

    for _ in range(int(max_iters)):
        to_fill = mask & (~known)
        if not to_fill.any():
            break

        val = depth_fg.clone()
        val[~known] = 0.0

        neigh_vals = neighborhood_stack(val)                # (H,W,9)
        neigh_known = neighborhood_stack(known.float())     # (H,W,9)

        cnt_nb = neigh_known.sum(dim=-1)                    # (H,W)
        fillable = to_fill & (cnt_nb >= min_count)
        if not fillable.any():
            break

        neigh_vals_masked = torch.where(neigh_known > 0, neigh_vals, torch.full_like(neigh_vals, float('nan')))

        neigh_med = torch.nanmedian(neigh_vals_masked, dim=-1).values  # (H,W)
        abs_dev = (neigh_vals_masked - neigh_med.unsqueeze(-1)).abs()
        neigh_mad = torch.nanmedian(abs_dev, dim=-1).values + eps      # (H,W)

        global_mad = core_mad if core_vals.numel() > 0 else torch.tensor(1.0, device=device)
        stable_enough = (neigh_mad <= mad_guard * global_mad)
        fillable = fillable & stable_enough
        if not fillable.any():
            break

        if use_bilateral:
            sigma_r = sigma_r_scale * neigh_mad  # (H,W)
            range_w = torch.exp(- ( (neigh_vals_masked - neigh_med.unsqueeze(-1)).abs() / (sigma_r.unsqueeze(-1) + eps) ))
            w = torch.where(neigh_known > 0, range_w, torch.zeros_like(range_w))
        else:
            w = None

        new_vals = torch.empty_like(depth_fg)
        new_vals[...] = float('nan')

        if use_bilateral:
            wsum = w.sum(dim=-1).clamp_min(eps)                 # (H,W)
            vws  = (w * torch.nan_to_num(neigh_vals_masked, nan=0.0)).sum(dim=-1)
            est  = vws / wsum
        else:
            est = neigh_med

        est = est.clamp(lo_clip, hi_clip)

        new_vals[fillable] = est[fillable]
        depth_fg[fillable] = new_vals[fillable]
        known = torch.isfinite(depth_fg)

    still_nan = mask & torch.isnan(depth_fg)
    if still_nan.any():
        depth_fg[still_nan] = depth[still_nan].clamp(lo_clip, hi_clip)


    if background.lower() == "keep":
        out = torch.where(mask, depth_fg, depth)
    elif background.lower() == "zero":
        out = torch.where(mask, torch.nan_to_num(depth_fg, nan=0.0), torch.zeros_like(depth))
    else:  # "nan"
        out = torch.where(mask, depth_fg, torch.full_like(depth, float("nan")))
    return out

def opencv_to_pytorch3d_points(opencv_points):
    """
    Convert points from OpenCV camera coordinate system to PyTorch3D camera coordinate system.
    
    Args:
        opencv_points: numpy array or torch tensor of shape (H, W, 3) or (..., 3)
                      Points in OpenCV format (x-right, y-down, z-forward)
    
    Returns:
        pytorch3d_points: points in PyTorch3D format (x-left, y-up, z-forward)
                         Same shape and type as input
    """
    # Handle both numpy arrays and torch tensors
    is_numpy = isinstance(opencv_points, np.ndarray)
    
    if is_numpy:
        # For numpy arrays
        pytorch3d_points = opencv_points.copy()
        # Flip x and y coordinates: (x, y, z) -> (-x, -y, z)
        pytorch3d_points[..., 0] = -opencv_points[..., 0]  # x: right -> left
        pytorch3d_points[..., 1] = -opencv_points[..., 1]  # y: down -> up
        # z remains the same (forward in both systems)
    else:
        # For torch tensors
        pytorch3d_points = opencv_points.clone()
        pytorch3d_points[..., 0] = -opencv_points[..., 0]  # x: right -> left
        pytorch3d_points[..., 1] = -opencv_points[..., 1]  # y: down -> up
        # z remains the same
    
    return pytorch3d_points

def save_tensor_as_image(image_tensor, filename):
    img_np = image_tensor.detach().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)

    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)

    img_pil = Image.fromarray(img_np)
    img_pil.save(filename)

def save_point_cloud_as_ply(pc, colors, output_path):
    import open3d as o3d

    # Convert the points and colors to Open3D format
    points_o3d = o3d.utility.Vector3dVector(pc.cpu().numpy())
    colors_o3d = o3d.utility.Vector3dVector(colors.cpu().numpy())

    # Create a point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = points_o3d
    point_cloud.colors = colors_o3d

    # Save the point cloud as a PLY file
    o3d.io.write_point_cloud(output_path, point_cloud)

def intrinsics_to_fov_opencv(intrinsics, image_size=(512, 512)):
    # Extract focal lengths (in normalized coordinates)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Get image dimensions
    height, width = image_size[0], image_size[1]

    # Convert normalized focal lengths to pixels
    fx_pixels = fx * width
    fy_pixels = fy * height

    fov_x_rad = 2 * torch.atan(width / (2 * fx_pixels))
    fov_y_rad = 2 * torch.atan(height / (2 * fy_pixels))

    fov_x_deg = torch.rad2deg(fov_x_rad)
    fov_y_deg = torch.rad2deg(fov_y_rad)

    return fov_x_deg, fov_y_deg, fx_pixels, fy_pixels

def remove_isolated_areas(mask, min_size, method='connected_components'):
    """
    Remove small isolated areas from a binary mask while preserving main areas.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Input binary mask of shape (512, 512) or any 2D shape
        Values should be 0 (background) and 255 (foreground) or 0 and 1
    min_size : int
        Minimum size (in pixels) for areas to keep. Areas smaller than this will be removed
    method : str
        Method to use: 'connected_components', 'morphology', or 'skimage'
    
    Returns:
    --------
    numpy.ndarray
        Refined mask with small isolated areas removed
    """
    
    # Ensure mask is binary (0 and 1)
    if mask.max() > 1:
        binary_mask = (mask > 127).astype(np.uint8)
    else:
        binary_mask = mask.astype(np.uint8)
    
    if method == 'connected_components':
        return _remove_isolated_cv2(binary_mask, min_size)
    elif method == 'morphology':
        return _remove_isolated_morphology(binary_mask, min_size)
    elif method == 'skimage':
        return _remove_isolated_skimage(binary_mask, min_size)
    else:
        raise ValueError("Method must be 'connected_components', 'morphology', or 'skimage'")


def _remove_isolated_cv2(binary_mask, min_size):
    """Remove isolated areas using OpenCV connected components."""
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Create output mask
    refined_mask = np.zeros_like(binary_mask)
    
    # Keep components that are large enough (skip background label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            refined_mask[labels == i] = 1
    
    return refined_mask


def _remove_isolated_morphology(binary_mask, min_size):
    """Remove isolated areas using morphological operations."""
    # Label connected components
    labeled_mask, num_features = ndimage.label(binary_mask)
    
    # Find sizes of each component
    component_sizes = ndimage.sum(binary_mask, labeled_mask, range(num_features + 1))
    
    # Create mask for components to keep
    mask_sizes = component_sizes < min_size
    remove_pixel = mask_sizes[labeled_mask]
    
    # Remove small components
    refined_mask = binary_mask.copy()
    refined_mask[remove_pixel] = 0
    
    return refined_mask


def _remove_isolated_skimage(binary_mask, min_size):
    """Remove isolated areas using scikit-image."""
    # Convert to boolean for skimage
    bool_mask = binary_mask.astype(bool)
    
    # Remove small objects
    cleaned_mask = remove_small_objects(bool_mask, min_size=min_size, connectivity=2)
    
    return cleaned_mask.astype(np.uint8)


def remove_isolated_areas_adaptive(mask, size_ratio=0.01):
    """
    Automatically determine the minimum size threshold based on the largest component.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Input binary mask
    size_ratio : float
        Ratio of the largest component size to use as minimum threshold
        
    Returns:
    --------
    numpy.ndarray
        Refined mask with small isolated areas removed
    """
    # Ensure mask is binary
    if mask.max() > 1:
        binary_mask = (mask > 127).astype(np.uint8)
    else:
        binary_mask = mask.astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    if num_labels <= 1:  # No foreground components
        return binary_mask
    
    # Find the largest component (excluding background)
    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background (label 0)
    largest_area = np.max(areas)
    
    # Set minimum size as a ratio of the largest component
    min_size = int(largest_area * size_ratio)
    min_size = max(min_size, 10)  # Ensure minimum threshold
    
    return _remove_isolated_cv2(binary_mask, min_size)


def intrinsics_to_fov_opencv(intrinsics, image_size=(512, 512)):
    # Extract focal lengths (in normalized coordinates)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Get image dimensions
    height, width = image_size[0], image_size[1]

    # Convert normalized focal lengths to pixels
    fx_pixels = fx * width
    fy_pixels = fy * height

    fov_x_rad = 2 * torch.atan(width / (2 * fx_pixels))
    fov_y_rad = 2 * torch.atan(height / (2 * fy_pixels))

    fov_x_deg = torch.rad2deg(fov_x_rad)
    fov_y_deg = torch.rad2deg(fov_y_rad)

    return fx_pixels, fy_pixels, fov_x_deg, fov_y_deg, fov_x_rad, fov_y_rad


def visualize_optical_flow_advanced(frames_folder, flows_npy_path, output_folder, 
                                   arrow_density=10, min_magnitude=1.0, dpi=300):
    
    flows = np.load(flows_npy_path)  # (143, 2, 240, 416)
    print(f"Loaded flows with shape: {flows.shape}")
    
    frame_files = sorted(glob.glob(os.path.join(frames_folder, "*.png")) + 
                        glob.glob(os.path.join(frames_folder, "*.jpg")))
    
    os.makedirs(output_folder, exist_ok=True)
    
    for i, frame_file in enumerate(frame_files):
        if i >= len(flows):
            break

        frame = cv2.imread(frame_file)
        frame_h, frame_w = frame.shape[:2]
        
        dx, dy = flows[i]
        flow_h, flow_w = dx.shape
        
        flow_tensor = torch.from_numpy(np.stack([dx, dy])).float()
        flow_tensor = torch.nn.functional.interpolate(
            flow_tensor.unsqueeze(0), 
            size=(frame_h, frame_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)  # [2, 480, 832]
        
        scale_h = frame_h / flow_h  # 2
        scale_w = frame_w / flow_w  # 2
        flow_tensor[0] *= scale_w  # x
        flow_tensor[1] *= scale_h  # y
        
        result = visualize_flow_arrows_only(
            flow_tensor, 
            (frame_h, frame_w),
            arrow_density=arrow_density,
            min_magnitude=min_magnitude,
            dpi=dpi
        )
        
        result_np = (result.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        output_path = os.path.join(output_folder, f"flow_vis_{i:04d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))
        
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(frame_files)} frames")
    
    print(f"Flow visualization completed! Saved to {output_folder}")



#  ffmpeg -y -framerate 10 -pattern_type glob -i 'result/sand_house/13-11_06-51-16/final_sim/flows/flow_vis_*.png' -c:v libx264 -pix_fmt yuv420p result/sand_house/13-11_06-51-16/final_sim/flows/flows_video_10fps.mp4   


def visualize_flow_arrows_only(flow, image_size, arrow_density=40, min_magnitude=1.0, dpi=100):
    
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    # flow: [2, H, W]
    # image_size: (H, W)
    flow = flow.permute(1, 2, 0).detach().cpu().numpy()
    h, w = image_size

    # use arrow_density to control the spacing between arrows
    spacing = arrow_density
    
    y, x = np.mgrid[0:h:spacing, 0:w:spacing]

    flow_y = flow[::spacing, ::spacing, 1]
    flow_y = 1 * flow_y    # matplotlib y-axis is upward
    flow_x = flow[::spacing, ::spacing, 0]
    
    # Calculate magnitude of flow vectors
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    
    # filter small flow magnitudes
    mask = magnitude > min_magnitude

    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])  # Make the axes occupy the whole figure
    
    # set pure white background
    ax.set_facecolor('white')
    
    # draw arrows, the length of the arrows varies with the flow size, but the size of the arrows is fixed
    ax.quiver(
        x[mask], y[mask],
        flow_x[mask], flow_y[mask],
        magnitude[mask],
        # key parameters adjustment:
        scale_units='xy',  # use xy coordinate units, let the length of the arrows directly correspond to the size of the flow
        angles='xy',       # use xy coordinate system for arrow angles
        scale=0.18,           # set to 1, let the length of the arrows directly correspond to the size of the flow vector
        cmap='jet',        # color mapping
        width=0.006,       # arrow line width (fixed)
        headwidth=18,       # arrow head width (fixed)
        headlength=12,      # arrow head length (fixed)
        headaxislength=6,  # arrow head axis length (fixed)
        minshaft=1,        # minimum arrow shaft length
        minlength=0,       # allow zero length arrows
    )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # flip y axis
    ax.set_axis_off()
    plt.margins(0, 0)
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    buf = canvas.buffer_rgba()
    ret = np.asarray(buf)
    
    plt.close(fig)
    
    ret = ret[:, :, :3] / 255.0
    
    ret = torch.from_numpy(ret.transpose(2, 0, 1)).float()

    return ret


def create_flow_arrows_demo(output_path="flow_arrows_demo.png", arrow_density=15):
    """
    create a demo showing how different flow sizes affect arrow lengths
    """
    
    # create a simple flow field for demo
    h, w = 200, 300
    
    # create flow vectors of different sizes
    flow = np.zeros((h, w, 2))
    
    # horizontal flow, intensity increases from left to right
    for i in range(w):
        strength = (i / w) * 20  # intensity from 0 to 20
        flow[:, i, 0] = strength  # x
    
    # add some vertical downward flow
    flow[h//3:2*h//3, :, 1] = 10
    
    # convert to tensor format
    flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).float()
    
    # visualize
    result = visualize_flow_arrows_only(
        flow_tensor, 
        (h, w),
        arrow_density=arrow_density,
        min_magnitude=0.5
    )
    
    # save
    result_np = (result.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))
    print(f"Demo saved to {output_path}")



def dilate_binary_mask(mask: np.ndarray, size=(512, 512), kernel_size=5, iterations=1):
    """
    Args:
        mask: np.ndarray, binary mask with values 0 or 1
        size: output image size (W, H)
        kernel_size: dilation kernel size (odd int)
        iterations: number of dilation iterations
    Returns:
        PIL.Image: dilated binary mask as RGB image with values 0 or 255
    """
    # Ensure mask is binary uint8 (0 or 255)
    mask = mask.detach().cpu().numpy()
    mask_uint8 = (mask.astype(np.uint8) * 255)

    # Resize to target size
    mask_resized = cv2.resize(mask_uint8, size, interpolation=cv2.INTER_NEAREST)

    # Apply dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_resized, kernel, iterations=iterations)

    # Convert to RGB and ensure it's still binary (0 or 255)
    dilated_binary = np.where(dilated > 0, 255, 0).astype(np.uint8)

    return dilated_binary

def smooth_segmentation_mask_255(
    mask: np.ndarray,
    blur_kernel_size: int = 15,
    blur_sigma: float = 5.0,
    threshold: int = 60,
    binary_output: bool = True,
    morph_close: bool = True,
    morph_kernel_size: int = 7,
    return_pil: bool = True
):
    """
    Smooth a 0-255 binary mask (uint8), soften edges to make blob-like shape.

    Args:
        mask (np.ndarray): Input mask with values 0 or 255, shape (H, W)
        blur_kernel_size (int): Size of Gaussian blur kernel
        blur_sigma (float): Sigma value for Gaussian blur
        threshold (int): Threshold value for binary output (0-255)
        binary_output (bool): If True, return 0/255 mask; if False, return soft mask
        morph_close (bool): Apply morphological closing to remove holes
        morph_kernel_size (int): Kernel size for morphological ops
        return_pil (bool): If True, return PIL image, else NumPy array

    Returns:
        PIL.Image or np.ndarray: Smoothed mask image (L mode)
    """
    assert mask.ndim == 2, "Mask must be 2D"
    assert np.issubdtype(mask.dtype, np.uint8), "Mask must be uint8"
    assert set(np.unique(mask)).issubset({0, 255}), "Mask must contain only 0 or 255"

    # Step 1: normalize to [0, 1]
    mask_norm = (mask / 255.0).astype(np.float32)

    # Step 2: blur
    blurred = cv2.GaussianBlur(mask_norm, (blur_kernel_size, blur_kernel_size), sigmaX=blur_sigma)

    # Step 3: threshold or scale
    if binary_output:
        mask_out = (blurred > (threshold / 255.0)).astype(np.uint8) * 255
    else:
        mask_out = np.clip(blurred * 255.0, 0, 255).astype(np.uint8)

    # Step 4: morphological closing (optional)
    if morph_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(mask_out).convert("L") if return_pil else mask_out


def render_mesh_with_occlusion_detection(vertices, faces, colors, camera, 
                                        image_size=512, device='cuda',
                                        depth_tolerance=0.01):
    """
    Render the mesh and detect occluded vertices.
    
    Args:
        vertices: vertex coordinates (N, 3)
        faces: face indices (F, 3)
        colors: vertex colors (N, 3)
        camera: PyTorch3D camera object
        image_size: rendered image size
        device: device
        depth_tolerance: depth tolerance for determining if a vertex is occluded
        
    Returns:
        mask: object mask
        depth_map: depth map
        occluded_vertices_mask: Boolean mask for occluded vertices (N,) - True means occluded
    """
    
    # Convert to tensor and move to device
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(faces, torch.Tensor):
        faces = torch.tensor(faces, dtype=torch.long)
    if not isinstance(colors, torch.Tensor):
        colors = torch.tensor(colors, dtype=torch.float32)
    
    vertices = vertices.to(device=device, dtype=torch.float32)
    faces = faces.to(device=device, dtype=torch.long)
    colors = colors.to(device=device, dtype=torch.float32)
    
    # Ensure colors are in the correct range [0, 1]
    if colors.max() > 1.0:
        colors = colors / 255.0
    
    # Add batch dimension
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)  # (1, N, 3)
    if faces.dim() == 2:
        faces = faces.unsqueeze(0)  # (1, F, 3)
    if colors.dim() == 2:
        colors = colors.unsqueeze(0)  # (1, N, 3)
    
    # Create mesh with vertex colors
    textures = TexturesVertex(verts_features=colors)
    mesh = Meshes(verts=vertices, faces=faces, textures=textures)
    
    # Set rasterization parameters
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
        max_faces_per_bin=None,
        perspective_correct=True,
    )
    
    # Create rasterizer
    rasterizer = MeshRasterizer(
        cameras=camera,
        raster_settings=raster_settings
    )
    
    # Create shader
    shader = SoftPhongShader(
        device=device,
        cameras=camera,
    )
    
    # Create renderer
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=False):
            # Get rasterized fragments for depth and mask
            fragments = rasterizer(mesh)
            
            # Render images
            rendered_images = renderer(mesh)
        
        # Extract components
        rendered_image = rendered_images[0, ..., :3]  # Remove batch dimension and alpha channel
        
        # Create mask from z-buffer (depth buffer)
        zbuf = fragments.zbuf[0, ..., 0]  # shape: (H, W)
        mask = zbuf != -1  # Convert to float mask (1 means object, 0 means background)
        
        # Create depth map
        depth_map = zbuf.clone()
        depth_map[zbuf == -1] = float('inf')  # Set background to infinity
        
        # Detect occluded vertices
        visible_vertices_mask, occluded_vertices_mask = detect_occluded_vertices(
            vertices[0], camera, depth_map, image_size, depth_tolerance, device
        )
    
    return mask, depth_map, occluded_vertices_mask


def detect_occluded_vertices(vertices, camera, depth_map, image_size, depth_tolerance, device):
    """
    Detect occluded vertices.
    
    Args:
        vertices: vertex coordinates (N, 3)
        camera: PyTorch3D camera object
        depth_map: rendered depth map (H, W)
        image_size: image size
        depth_tolerance: depth tolerance
        device: device
        
    Returns:
        visible_vertices_mask: Boolean mask for visible vertices (N,) - True means visible
        occluded_vertices_mask: Boolean mask for occluded vertices (N,) - True means occluded
    """
    # Early exit for empty vertices
    if vertices.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=device), torch.zeros((0,), dtype=torch.bool, device=device)

    # Derive image size from depth map to avoid mismatch
    image_h, image_w = depth_map.shape[-2], depth_map.shape[-1]

    # Project vertices to screen coordinates
    screen_coords = camera.transform_points(vertices.unsqueeze(0))[0]  # (N, 3)

    # Filter out invalid projections
    valid_mask = torch.isfinite(screen_coords).all(dim=1)
    if valid_mask.sum() == 0:
        return torch.zeros((vertices.shape[0],), dtype=torch.bool, device=device), torch.ones((vertices.shape[0],), dtype=torch.bool, device=device)

    pixel_coords = torch.nan_to_num(screen_coords[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    vertex_depths = torch.nan_to_num(1 / screen_coords[:, 2], nan=float("inf"), posinf=float("inf"), neginf=float("inf"))

    pixel_coords_x = torch.clamp(pixel_coords[:, 0], 0, image_w - 1)
    pixel_coords_y = torch.clamp(pixel_coords[:, 1], 0, image_h - 1)
    pixel_x = image_w - 1 - pixel_coords_x.long()
    pixel_y = image_h - 1 - pixel_coords_y.long()

    # Get the depth value for each vertex's corresponding pixel position
    pixel_depths = depth_map[pixel_y, pixel_x]

    # Determine if the vertex is visible
    depth_diff = torch.abs(vertex_depths - pixel_depths)

    # Handle background pixels (case where depth is infinity)
    is_background = pixel_depths == float('inf')

    visible_vertices_mask = (depth_diff < depth_tolerance) & (~is_background) & valid_mask
    occluded_vertices_mask = ~visible_vertices_mask

    return visible_vertices_mask, occluded_vertices_mask


def create_occluded_submesh(vertices, faces, colors, occluded_vertices_mask):
    
    device = vertices.device

    occluded_faces_mask, occluded_faces, _, _ = extract_occluded_faces(faces, occluded_vertices_mask)
   
    assert occluded_faces_mask.any(), "No occluded faces found"
    
    # Get all the used vertices in the occluded faces
    used_vertices = torch.unique(occluded_faces.flatten())
    
    # Create a vertex mapping: old_index -> new_index
    vertex_mapping = torch.full((vertices.shape[0],), -1, dtype=torch.long, device=device)
    vertex_mapping[used_vertices] = torch.arange(len(used_vertices), device=device)
    
    # Extract submesh vertices and colors
    submesh_vertices = vertices[used_vertices]
    submesh_colors = colors[used_vertices]
    
    # Remap face indices
    submesh_faces = vertex_mapping[occluded_faces]
    
    return submesh_vertices, submesh_faces, submesh_colors


def extract_occluded_faces(faces, occluded_vertices_mask):
    """
    Extract faces that contain occluded vertices.
    
    Args:
        faces: face indices (F, 3)
        occluded_vertices_mask: Boolean mask for occluded vertices (N,) - True means occluded
        
    Returns:
        occluded_faces_mask: Boolean mask for occluded faces (F,) - True means face contains occluded vertices
        occluded_faces: indices of occluded faces
        partially_occluded_faces_mask: mask of partially occluded faces (only some vertices are occluded)
        fully_occluded_faces_mask: mask of fully occluded faces (all vertices are occluded)
    """
    
    device = faces.device
    num_faces = faces.shape[0]
    
    # Check for each face whether each vertex is occluded
    face_vertex_occluded = occluded_vertices_mask[faces]  # (F, 3) - occlusion status of each vertex per face
    
    # Count how many vertices in each face are occluded
    num_occluded_per_face = face_vertex_occluded.sum(dim=1)  # (F,)
    
    # Different types of occluded faces
    no_occluded_faces_mask = num_occluded_per_face == 0           # No occluded vertices
    partially_occluded_faces_mask = (num_occluded_per_face > 0) & (num_occluded_per_face < 3)  # Some vertices occluded
    fully_occluded_faces_mask = num_occluded_per_face == 3        # All vertices occluded
    
    # Faces containing any occluded vertex
    occluded_faces_mask = num_occluded_per_face > 0
    occluded_faces = faces[occluded_faces_mask]
    
    return (occluded_faces_mask, occluded_faces, 
            partially_occluded_faces_mask, fully_occluded_faces_mask)
