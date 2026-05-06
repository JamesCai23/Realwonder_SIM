import argparse
import torch
import os
import gc
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize
import cv2
import numpy as np
import torch.nn.functional as F
import random
from pathlib import Path
from datetime import datetime
from simulation.genesis_simulator import DiffSim
from simulation.image23D.single_view_reconstructor import SingleViewReconstructor
from simulation.image23D.noise_warp.make_warped_noise import NoiseWarper
from simulation.image23D.inpainter import FluxInpainter
from simulation.image23D.segmenter import SegmentAnything3Segmenter
from simulation.image23D.mesh_generator import Sam3DMeshGenerator
from moge.model.v1 import MoGeModel
from simulation.utils import save_video_from_pil, save_gif_from_image_folder, resize_and_crop_pil, visualize_optical_flow_advanced


_PRELOADED_RUNTIME_KEYS = set()
_SHARED_NOISE_WARPER = None


def _resolve_runtime_devices(config):
    device = str(config.get("device", "cuda:0"))
    sam3_device = str(config.get("sam3_device", device))
    inpaint_device = str(config.get("inpaint_device", device))
    noise_device = str(config.get("noise_device", device))
    return device, sam3_device, inpaint_device, noise_device


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _cuda_housekeeping():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _reset_torch_default_device_to_cpu():
    # Genesis may set a global CUDA default device; reset it per case to avoid
    # leaking device state into next-case inpainting/tokenization.
    try:
        torch.set_default_device("cpu")
    except Exception:
        pass


def _cuda_synchronize_all_devices():
    if not torch.cuda.is_available():
        return
    device_count = torch.cuda.device_count()
    for idx in range(device_count):
        torch.cuda.synchronize(idx)


def _maybe_release_inpaint_cache(config):
    release_inpaint = _env_flag(
        "REALWONDER_RELEASE_INPAINT_BEFORE_NOISE",
        bool(config.get("release_inpaint_before_noise", False)),
    )
    if not release_inpaint:
        return

    inpaint_device = str(config.get("inpaint_device", config.get("device", "cuda:0")))
    try:
        FluxInpainter.clear_cache(device=inpaint_device)
        print(f"[case_simulation] Released inpaint cache on device={inpaint_device}")
    except Exception as e:
        print(f"[WARN] Failed to release inpaint cache on {inpaint_device}: {e}")

    _cuda_housekeeping()


def _get_shared_noise_warper():
    global _SHARED_NOISE_WARPER
    if _SHARED_NOISE_WARPER is None:
        _SHARED_NOISE_WARPER = NoiseWarper()
    return _SHARED_NOISE_WARPER


def preload_case_simulation_models(config):
    """Preload heavyweight models once per process/device tuple to avoid per-case reload."""
    device, sam3_device, inpaint_device, _ = _resolve_runtime_devices(config)
    preload_inpaint = _env_flag(
        "REALWONDER_PRELOAD_INPAINT",
        bool(config.get("preload_inpaint", True)),
    )
    runtime_key = (device, sam3_device, inpaint_device, preload_inpaint)
    if runtime_key in _PRELOADED_RUNTIME_KEYS:
        return

    # SAM3 segmentation model
    try:
        SegmentAnything3Segmenter({"all_object_names": ["object"]}, device=sam3_device)
    except Exception as e:
        print(f"[WARN] SAM3 preload skipped: {e}")

    # SAM3D mesh model
    try:
        Sam3DMeshGenerator(config={}, device=device)
    except Exception as e:
        print(f"[WARN] SAM3D preload skipped: {e}")

    # FLUX inpainting pipeline
    if preload_inpaint:
        try:
            FluxInpainter(device=inpaint_device)
        except Exception as e:
            print(f"[WARN] Flux inpaint preload skipped: {e}")
    else:
        print("[case_simulation] Skip inpaint preload (REALWONDER_PRELOAD_INPAINT=0)")

    # MoGe depth model (cache key matches SingleViewReconstructor)
    moge_model_path = "/home/lff/bigdata1/huggingface/moge-vitl/model.pt"
    moge_cache_key = (moge_model_path, str(device))
    try:
        if moge_cache_key not in SingleViewReconstructor._MOGE_CACHE:
            moge_model = MoGeModel.from_pretrained(moge_model_path).to(device)
            moge_model.eval()
            SingleViewReconstructor._MOGE_CACHE[moge_cache_key] = moge_model
            print(f"[case_simulation] Preloaded MoGe model: {moge_model_path}")
        else:
            print(f"[case_simulation] Reusing preloaded MoGe model: {moge_model_path}")
    except Exception as e:
        print(f"[WARN] MoGe preload skipped: {e}")

    _PRELOADED_RUNTIME_KEYS.add(runtime_key)
    print(
        "[case_simulation] Runtime preload ready "
        f"(device={device}, sam3_device={sam3_device}, inpaint_device={inpaint_device})"
    )

def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def process_simulated_results(
    input_image,
    raw_video_frames,
    points_masks,
    mesh_masks,
    crop_start=176,
    output_size=(832, 480),
    preserve_aspect=True,
):
    input_image = resize_and_crop_pil(
        input_image,
        crop_start,
        target_size=output_size,
        preserve_aspect=preserve_aspect,
    )
    raw_video_frames = [
        resize_and_crop_pil(
            frame,
            crop_start,
            target_size=output_size,
            preserve_aspect=preserve_aspect,
        )
        for frame in raw_video_frames
    ]
    points_masks = preprocess_masks_downsample(points_masks, output_size=output_size, preserve_aspect=preserve_aspect)
    mesh_masks = preprocess_masks_downsample(mesh_masks, output_size=output_size, preserve_aspect=preserve_aspect)

    return input_image, raw_video_frames, points_masks, mesh_masks

def preprocess_masks_downsample(masks, output_size=(832, 480), preserve_aspect=True):
    '''
    input: list of numpy array (512, 512, 1)
    output: 
    '''
    num_masks = len(masks)
    masks = torch.stack(masks, dim=0).squeeze(-1)
    target_w, target_h = output_size
    masks = masks.unsqueeze(1).float()

    if preserve_aspect:
        src_h, src_w = masks.shape[-2], masks.shape[-1]
        scale = min(target_h / max(src_h, 1), target_w / max(src_w, 1))
        resized_h = max(1, int(round(src_h * scale)))
        resized_w = max(1, int(round(src_w * scale)))
        resized_masks = F.interpolate(masks, size=(resized_h, resized_w), mode='bilinear', align_corners=False)

        pad_h = target_h - resized_h
        pad_w = target_w - resized_w
        pad_top = max(pad_h // 2, 0)
        pad_bottom = max(pad_h - pad_top, 0)
        pad_left = max(pad_w // 2, 0)
        pad_right = max(pad_w - pad_left, 0)
        processed_masks = F.pad(resized_masks, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
    else:
        square_size = max(target_w, target_h)
        resized_masks = F.interpolate(masks, size=(square_size, square_size), mode='bilinear', align_corners=False)
        start_y = (square_size - target_h) // 2
        start_x = (square_size - target_w) // 2
        processed_masks = resized_masks[:, :, start_y:start_y + target_h, start_x:start_x + target_w]

    latent_h = target_h // 8
    latent_w = target_w // 8
    masks_downsampled = F.interpolate(processed_masks.float(), size=(latent_h, latent_w), mode='bilinear', align_corners=False).squeeze(1)
    time_averaged_masks = []
    for i in range(0, num_masks, 4):
        time_averaged_masks.append(masks_downsampled[i : i + 4, :, :].mean(dim=0, keepdim=True))
    masks_downsampled = torch.cat(time_averaged_masks, dim=0)
    masks_downsampled = masks_downsampled > 0.5
    return masks_downsampled # torch.Size([12, 60, 104])

def run_case_simulation(config_path, output_dir=None):

    config = OmegaConf.load(config_path)

    _reset_torch_default_device_to_cpu()

    # Allow runtime device override without regenerating config.yaml.
    env_device = os.environ.get("REALWONDER_DEVICE")
    env_sam3_device = os.environ.get("REALWONDER_SAM3_DEVICE")
    env_inpaint_device = os.environ.get("REALWONDER_INPAINT_DEVICE")
    env_noise_device = os.environ.get("REALWONDER_NOISE_DEVICE")
    if env_device:
        config['device'] = env_device
    if env_sam3_device:
        config['sam3_device'] = env_sam3_device
    if env_inpaint_device:
        config['inpaint_device'] = env_inpaint_device
    if env_noise_device:
        config['noise_device'] = env_noise_device

    preload_case_simulation_models(config)

    if output_dir is not None:
        output_folder = os.path.join(config['output_folder'], output_dir)
    else:
        timestamp = datetime.now().strftime("%d-%m_%H-%M-%S")
        output_folder = os.path.join(config['output_folder'], timestamp)
    os.makedirs(output_folder, exist_ok=True)
    config['output_folder'] = output_folder
    debug = config.get('debug', False)

    if debug:
        debug_config_save_path = os.path.join(config['output_folder'], "config.yaml")
        OmegaConf.save(config, debug_config_save_path)

    device = torch.device("cuda")
    set_seed(config['seed'])

    torch.set_grad_enabled(False)
    input_image = Image.open(os.path.join(config['data_path'], 'input.png')).convert('RGB')

    genesis_simulator = DiffSim(config)
    raw_video_frames, points_masks, mesh_masks = genesis_simulator.simulation_pc_render()

    output_width = int(config.get('output_width', 832))
    output_height = int(config.get('output_height', 480))
    preserve_aspect_ratio = bool(config.get('preserve_aspect_ratio', True))
    crop_start = int(config.get('crop_start', 176))

    input_image, video_frames, points_masks_downsampled, mesh_masks_downsampled = process_simulated_results(
        input_image,
        raw_video_frames,
        points_masks,
        mesh_masks,
        crop_start=crop_start,
        output_size=(output_width, output_height),
        preserve_aspect=preserve_aspect_ratio,
    )
    del raw_video_frames
    del points_masks
    del mesh_masks

    final_sim_folder = os.path.join(output_folder, "final_sim")
    os.makedirs(final_sim_folder, exist_ok=True)

    config_save_path = os.path.join(final_sim_folder, "config.yaml")
    OmegaConf.save(config, config_save_path)

    noise_warper = _get_shared_noise_warper()
    optical_flows = genesis_simulator.svr.optical_flow

    optical_flows = np.array(optical_flows)[..., :2]  # shape (71, 512, 512, 2)
    optical_flows = np.transpose(optical_flows, (0, 3, 1, 2))  # shape (71, 2, 512, 512)

    if debug:
        np.save(os.path.join(final_sim_folder, "flows.npy"), optical_flows)

    # save the simulation results
    frame_folder = os.path.join(final_sim_folder, "frames")
    os.makedirs(frame_folder, exist_ok=True)
    for i, frame in enumerate(video_frames):
        frame_path = os.path.join(frame_folder, f"frame_{i:04d}.png")
        frame.save(frame_path)

    if debug:
        visualize_optical_flow_advanced(frame_folder, os.path.join(final_sim_folder, "flows.npy"), os.path.join(final_sim_folder, "optical_flow_viz"), arrow_density=30)

    # Route noise generation to dedicated device when provided.
    noise_device = config.get('noise_device', config.get('device', 'cuda:0'))
    _maybe_release_inpaint_cache(config)
    del genesis_simulator
    _cuda_housekeeping()

    # warped_noise = noise_warper.process(optical_flows, final_sim_folder, crop_start=config['crop_start'], input_flow=True, debug=debug)
    warped_noise = noise_warper.process(video_frames, final_sim_folder, crop_start=config['crop_start'], input_flow=False, debug=debug, device=noise_device)
    _cuda_synchronize_all_devices()

    points_masks_path = os.path.join(final_sim_folder, "points_masks_downsampled.pt")
    torch.save(points_masks_downsampled, points_masks_path)
    mesh_masks_path = os.path.join(final_sim_folder, "mesh_masks_downsampled.pt")
    torch.save(mesh_masks_downsampled, mesh_masks_path)

    video_path = os.path.join(final_sim_folder, "simulation.mp4")
    save_video_from_pil(video_frames, video_path, fps=24)

    input_image_path = os.path.join(final_sim_folder, "resized_input_image.png")
    input_image.save(input_image_path)

    prompt_txt_path = os.path.join(final_sim_folder, "prompt.txt")
    with open(prompt_txt_path, "w") as f:
        f.write(config['vgen_prompt'])

    _reset_torch_default_device_to_cpu()

    return final_sim_folder

def main():     
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--output_dir", default=None, type=str, help="Path to the output folder")
    args = parser.parse_args()
    run_case_simulation(args.config_path, output_dir=args.output_dir)

if __name__ == "__main__":
    main()