"""Utility functions for video generation."""

import json
import random
from abc import abstractmethod, ABC
from pathlib import Path

import numpy as np
import torch
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset


# =============================================================================
# Seed and Config Utils
# =============================================================================

def set_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def apply_config_overrides(config, unknown_args, verbose=True):
    """Parse command line arguments and apply them as config overrides."""
    config_overrides = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            key = arg[2:]
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                value = unknown_args[i + 1]
                try:
                    if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                        value = int(value)
                    elif '.' in value:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    elif value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                except ValueError:
                    pass
                config_overrides[key] = value
                i += 2
            else:
                config_overrides[key] = True
                i += 1
        else:
            i += 1

    if config_overrides:
        if verbose:
            print("Config overrides:")
        for key, value in config_overrides.items():
            override_dict = {}
            keys = key.split('.')
            current = override_dict
            for k in keys[:-1]:
                current[k] = {}
                current = current[k]
            current[keys[-1]] = value
            config = OmegaConf.merge(config, OmegaConf.create(override_dict))
            if verbose:
                print(f"  {key} = {value}")
    else:
        if verbose:
            print("No config overrides")
    return config


# =============================================================================
# Dataset
# =============================================================================

def extract_subdim(tensor: torch.Tensor, select_dim: int, return_complement: bool = True, channel_dim: int = 1):
    """Randomly select a subset of channels and optionally return the complement."""
    num_channels = tensor.shape[channel_dim]
    if select_dim > num_channels:
        raise ValueError(f"select_dim ({select_dim}) cannot exceed num_channels ({num_channels})")

    perm = torch.randperm(num_channels, device=tensor.device)
    selected_idx = perm[:select_dim]
    complement_idx = perm[select_dim:]

    indexers = [slice(None)] * tensor.dim()
    indexers[channel_dim] = selected_idx
    selected = tensor[tuple(indexers)]

    if not return_complement:
        return selected

    indexers[channel_dim] = complement_idx
    complement = tensor[tuple(indexers)]
    return selected, complement


class TextImagePairDataset(Dataset):
    """Dataset for text-image pairs."""

    def __init__(self, data_dir, transform=None, eval_first_n=-1, pad_to_multiple_of=None):
        self.transform = transform
        data_dir = Path(data_dir)

        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        aspect_ratio = metadata_path.stem.split('_')[-1]

        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


# =============================================================================
# Noise Processing Functions
# =============================================================================

def blend_noise(noise_background, noise_foreground, alpha):
    """Variance-preserving blend of two noise tensors/arrays."""
    denominator = (alpha ** 2 + (1 - alpha) ** 2) ** 0.5
    return (noise_foreground * alpha + noise_background * (1 - alpha)) / denominator


def mix_new_noise(noise, alpha):
    """Blend existing noise with freshly sampled Gaussian noise."""
    if isinstance(noise, torch.Tensor):
        return blend_noise(noise, torch.randn_like(noise), alpha)
    elif isinstance(noise, np.ndarray):
        return blend_noise(noise, np.random.randn(*noise.shape), alpha)
    else:
        raise TypeError(f"Unsupported input type: {type(noise)}. Expected PyTorch Tensor or NumPy array.")


def _temporal_resize_nearest(noise, target_length):
    """Resize noise temporally using nearest neighbor interpolation."""
    if noise.shape[0] == target_length:
        return noise
    indices = torch.linspace(0, noise.shape[0] - 1, steps=target_length, device=noise.device)
    indices = torch.clamp(indices.round().long(), 0, noise.shape[0] - 1)
    return torch.index_select(noise, 0, indices)


def _temporal_segment_indices(total_frames, target_length):
    """Calculate segment boundaries for temporal downsampling."""
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    boundaries = torch.linspace(0, total_frames, steps=target_length + 1, device='cpu')
    boundaries = boundaries.round().long()
    boundaries[-1] = total_frames
    return boundaries


def downsamp_mean(noise, target_length):
    """Downsample noise by averaging segments."""
    boundaries = _temporal_segment_indices(noise.shape[0], target_length)
    segments = []
    for i in range(target_length):
        start = int(boundaries[i].item())
        end = int(boundaries[i + 1].item())
        if end <= start:
            end = min(start + 1, noise.shape[0])
        segment = noise[start:end]
        segments.append(segment.mean(dim=0, keepdim=True))
    return torch.cat(segments, dim=0)


def normalized_noises(noises):
    """Normalize noise tensors."""
    return torch.stack([x / (x.std(dim=1, keepdim=True) + 1e-12) for x in noises])


def get_downtemp_noise(noise, target_length, mode="nearest"):
    """Downsample noise temporally using specified mode."""
    assert mode in {"nearest", "blend", "blend_norm", "randn"}, mode
    if mode == "nearest":
        return _temporal_resize_nearest(noise, target_length)
    if mode == "blend":
        return downsamp_mean(noise, target_length)
    if mode == "blend_norm":
        return normalized_noises(downsamp_mean(noise, target_length))
    if mode == "randn":
        return torch.randn(target_length, *noise.shape[1:], device=noise.device, dtype=noise.dtype)
    raise AssertionError("unreachable")


# =============================================================================
# Noise Loading Utilities
# =============================================================================

def load_noise(
    noise_path: str,
    target_frames: int = 21,
    channel_dim: int = 16,
    downsample_mode: str = "nearest",
    eval_degradation: float = 0.0,
):
    """Load and prepare structured noise from a .npy file.

    Args:
        noise_path: Path to noise.npy file with shape [T, H, W, C] = [81, 60, 104, noise_dim]
        target_frames: Number of target frames after downsampling
        channel_dim: Number of channels for structured noise (rest goes to SDE noise)
        downsample_mode: Temporal downsampling mode ('nearest', 'blend', etc.)
        eval_degradation: Blend ratio with random noise (0.0 = pure structured noise)

    Returns:
        dict with keys: structured_noise, structured_noise_sde, degradation
    """
    noise_path = Path(noise_path)
    if not noise_path.exists():
        raise FileNotFoundError(f"Noise file not found: {noise_path}")

    noise_array = np.load(noise_path, allow_pickle=False)
    if noise_array.ndim != 4:
        raise ValueError(f"Noise array must be 4D, got {noise_array.shape}")

    noise_tensor = torch.from_numpy(noise_array).to(torch.float32)

    # Ensure tensor layout is [frames, channels, height, width]
    noise_tensor = noise_tensor.permute(0, 3, 1, 2).contiguous()

    # Temporal downsampling
    noise_tensor = get_downtemp_noise(noise_tensor, target_frames, mode=downsample_mode)

    # Blend with random noise if degradation > 0
    if eval_degradation > 0.0:
        noise_tensor = mix_new_noise(noise_tensor, eval_degradation)

    # Split into structured noise and SDE noise (deterministic slicing, not random)
    structured_noise = noise_tensor[:, :channel_dim]
    if noise_tensor.shape[1] > channel_dim:
        structured_noise_sde = noise_tensor[:, channel_dim:]
    else:
        structured_noise_sde = None

    return {
        "structured_noise": structured_noise,
        "structured_noise_sde": structured_noise_sde,
        "degradation": eval_degradation,
    }


def load_prompt(prompt_path: str) -> str:
    """Load prompt from a text file."""
    prompt_path = Path(prompt_path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_first_frame(image_path: str, height: int = 480, width: int = 832):
    """Load and preprocess the first frame image.

    Args:
        image_path: Path to the first frame image
        height: Target height
        width: Target width

    Returns:
        Preprocessed image tensor
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height))
    return preprocess_image(image).squeeze(0)


# =============================================================================
# Image Processing
# =============================================================================

def preprocess_image(image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
    """Transform a PIL.Image to torch.Tensor."""
    image = torch.Tensor(np.array(image, dtype=np.float32))
    image = image.to(dtype=torch_dtype or torch_dtype, device=device or device)
    image = image * ((max_value - min_value) / 255) + min_value
    image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
    return image


class PipelineUnit:
    """Base class for pipeline processing units."""

    def __init__(self, seperate_cfg: bool = False, take_over: bool = False,
                 input_params: tuple = None, input_params_posi: dict = None,
                 input_params_nega: dict = None, onload_model_names: tuple = None):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def process(self, pipe, inputs: dict, positive=True, **kwargs) -> dict:
        raise NotImplementedError("`process` is not implemented.")


class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    """CLIP image embedder for WAN video pipeline."""

    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, image_encoder, input_image, end_image, height, width, device, torch_dtype):
        image = input_image.to(device)
        clip_context = image_encoder.encode_image([image])
        clip_context = clip_context.to(dtype=torch_dtype, device=device)
        return {"clip_feature": clip_context}


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    """VAE image embedder for WAN video pipeline."""

    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width"),
            onload_model_names=("vae",)
        )

    def process(self, vae, input_image, end_image, num_frames, height, width, device, torch_dtype):
        image = input_image.to(device)
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=device)
        msk[:, 1:] = 0
        vae_input = torch.concat([image.transpose(0, 1),
                                  torch.zeros(3, num_frames - 1, height, width).to(image.device)], dim=1)
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        y = vae.encode([vae_input.to(dtype=torch_dtype, device=device)], device=device,
                       tiled=True, tile_size=(30, 52), tile_stride=(15, 26))[0]
        y = y.to(dtype=torch_dtype, device=device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=torch_dtype, device=device)
        return {"y": y}


# =============================================================================
# Scheduler
# =============================================================================

class SchedulerInterface(ABC):
    """Base class for diffusion noise schedule."""
    alphas_cumprod: torch.Tensor

    @abstractmethod
    def add_noise(self, clean_latent: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor):
        pass

    def convert_x0_to_noise(self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = x0.dtype
        x0, xt, alphas_cumprod = map(lambda x: x.double().to(x0.device), [x0, xt, self.alphas_cumprod])
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        noise_pred = (xt - alpha_prod_t ** 0.5 * x0) / beta_prod_t ** 0.5
        return noise_pred.to(original_dtype)

    def convert_noise_to_x0(self, noise: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = noise.dtype
        noise, xt, alphas_cumprod = map(lambda x: x.double().to(noise.device), [noise, xt, self.alphas_cumprod])
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        x0_pred = (xt - beta_prod_t ** 0.5 * noise) / alpha_prod_t ** 0.5
        return x0_pred.to(original_dtype)

    def convert_velocity_to_x0(self, velocity: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = velocity.dtype
        velocity, xt, alphas_cumprod = map(lambda x: x.double().to(velocity.device), [velocity, xt, self.alphas_cumprod])
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        x0_pred = (alpha_prod_t ** 0.5) * xt - (beta_prod_t ** 0.5) * velocity
        return x0_pred.to(original_dtype)


class FlowMatchScheduler:
    """Flow matching scheduler for diffusion models."""

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0,
                 sigma_max=1.0, sigma_min=0.003 / 1.002, inverse_timesteps=False,
                 extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())

    def step(self, model_output, timestep, sample, to_final=False):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples, noise, timestep):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0)
        return self.linear_timesteps_weights[timestep_id]
