"""WAN model wrappers for video generation."""

import types
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm
from peft import LoraConfig, inject_adapter_in_model
from safetensors import safe_open

from vidgen.utils import SchedulerInterface, FlowMatchScheduler
from vidgen.memory import AutoWrappedLinear
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.clip import clip_xlm_roberta_vit_h_14
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel


# LoRA Configuration
gwtf_lora_config = LoraConfig(
    r=2048,
    lora_alpha=2048,
    target_modules=["k", "o", "q", "v", "ffn.0", "ffn.2"]
)

type_dict = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    """Load state dict from safetensors file."""
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


class GeneralLoRALoader:
    """General LoRA loader for model weights."""

    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype

    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict

    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        updated_num = 0
        lora_name_dict = self.get_name_dict(state_dict_lora)
        for name, module in model.named_modules():
            if name in lora_name_dict:
                weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=self.device, dtype=self.torch_dtype)
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)
                state_dict = module.state_dict()
                state_dict["weight"] = state_dict["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict)
                updated_num += 1
        print(f"{updated_num} tensors are updated by LoRA.")
        print(f"The number of LoRA parameters is {len(lora_name_dict)}.")


class WanTextEncoder(torch.nn.Module):
    """WAN text encoder wrapper."""

    def __init__(self):
        super().__init__()
        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("wan_models/Wan2.1-Fun-V1.1-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )
        self.tokenizer = HuggingfaceTokenizer(
            name="wan_models/Wan2.1-Fun-V1.1-1.3B-InP/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        try:
            return self.text_encoder.token_embedding.weight.device
        except StopIteration:
            return torch.device("cuda", torch.cuda.current_device())

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        target_device = self.device
        ids = ids.to(target_device)
        mask = mask.to(target_device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0
        return {"prompt_embeds": context}


class WanImageEncoder(torch.nn.Module):
    """WAN image encoder wrapper."""

    def __init__(self):
        super().__init__()
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device="cpu")
        self.model = self.model.eval().requires_grad_(False)
        checkpoint_path = "wan_models/Wan2.1-Fun-V1.1-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        logging.info(f'loading {checkpoint_path}')
        # Using weights_only=True significantly speeds up loading and improves security
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)

    def encode_image(self, videos):
        size = (self.model.image_size,) * 2
        videos = torch.cat([
            F.interpolate(u, size=size, mode='bicubic', align_corners=False) for u in videos
        ])
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))
        dtype = next(iter(self.model.visual.parameters())).dtype
        videos = videos.to(dtype)
        out = self.model.visual(videos, use_31_block=True)
        return out


class WanVAEWrapper(torch.nn.Module):
    """WAN VAE wrapper for encoding/decoding."""

    def __init__(self):
        super().__init__()
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.model = _video_vae(
            pretrained_path="wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = [self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel]
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def cached_encode_to_latent(self, pixel: torch.Tensor, is_first: bool = True) -> torch.Tensor:
        """Like encode_to_latent but uses cached_encode() to preserve caches across blocks.

        Args:
            pixel: [B, C, T, H, W] input pixel tensor.
            is_first: True for the first block (fresh cache, 1+4N chunking),
                      False for subsequent blocks (keep cache, 4N chunking).
        """
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = [self.model.cached_encode(u.unsqueeze(0), scale, is_first=is_first).float().squeeze(0) for u in pixel]
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"
        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        decode_function = self.model.cached_decode if use_cache else self.model.decode
        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanVideoVAE(nn.Module):
    """WAN Video VAE with tiled encoding/decoding support."""

    def __init__(self, z_dim=16):
        super().__init__()
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]
        self.model = _video_vae(
            pretrained_path="wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)
        self.upsampling_factor = 8
        self.z_dim = z_dim

    def scale_to(self, device):
        self.scale = [v.to(device) for v in self.scale]

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])
        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)
        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    def tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = device
        computation_device = device
        out_T = (T + 3) // 4
        weight = torch.zeros((1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
                             dtype=video.dtype, device=data_device)
        values = torch.zeros((1, self.z_dim, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
                             dtype=video.dtype, device=data_device)

        for h, h_, w, w_ in tqdm(tasks, desc="VAE encoding"):
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale).to(data_device)
            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=((size_h - stride_h) // self.upsampling_factor,
                              (size_w - stride_w) // self.upsampling_factor)
            ).to(dtype=video.dtype, device=data_device)
            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[:, :, :, target_h:target_h + hidden_states_batch.shape[3],
                   target_w:target_w + hidden_states_batch.shape[4]] += hidden_states_batch * mask
            weight[:, :, :, target_h:target_h + hidden_states_batch.shape[3],
                   target_w:target_w + hidden_states_batch.shape[4]] += mask
        values = values / weight
        return values

    def single_encode(self, video, device):
        video = video.to(device)
        return self.model.encode(video, self.scale)

    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        self.scale_to(device)
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size = (tile_size[0] * self.upsampling_factor, tile_size[1] * self.upsampling_factor)
                tile_stride = (tile_stride[0] * self.upsampling_factor, tile_stride[1] * self.upsampling_factor)
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        return torch.stack(hidden_states)


class WanDiffusionWrapper(torch.nn.Module):
    """WAN diffusion model wrapper for T2V."""

    def __init__(self, model_name="Wan2.1-Fun-V1.1-1.3B-InP", timestep_shift=8.0, is_causal=False,
                 local_attn_size=-1, sink_size=0, lora=False, lora_ckpt_path=None,
                 lora_alpha=1.0, upcast_dtype=None, hotload=False, lora_training=True):
        super().__init__()

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")

        if lora:
            self._setup_lora(model_name, lora_ckpt_path, lora_alpha, upcast_dtype, lora_training)

        self.model.eval()
        self.uniform_timestep = not is_causal
        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.seq_len = 32760
        self.post_init()

    def _setup_lora(self, model_name, lora_ckpt_path, lora_alpha, upcast_dtype, lora_training):
        print(f"Injecting LoRA into {model_name}")
        if lora_training:
            self.model = inject_adapter_in_model(gwtf_lora_config, self.model)

        if upcast_dtype is not None and upcast_dtype != "float32":
            upcast_dtype = type_dict[upcast_dtype]
            for param in self.model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)

        if lora_ckpt_path is not None:
            lora = load_state_dict_from_safetensors(lora_ckpt_path, torch_dtype=upcast_dtype, device=self.model.device)
            loader = GeneralLoRALoader(torch_dtype=upcast_dtype, device=self.model.device)
            loader.load(self.model, lora, alpha=lora_alpha)

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def forward(self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
                timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
                crossattn_cache: Optional[List[dict]] = None, current_start: Optional[int] = None,
                clean_x: Optional[torch.Tensor] = None, aug_t: Optional[torch.Tensor] = None,
                cache_start: Optional[int] = None, **kwargs) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        input_timestep = timestep[:, 0] if self.uniform_timestep else timestep

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len, kv_cache=kv_cache,
                crossattn_cache=crossattn_cache, current_start=current_start,
                cache_start=cache_start
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds, seq_len=self.seq_len
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        self.get_scheduler()


class WanI2VDiffusionWrapper(torch.nn.Module):
    """WAN diffusion model wrapper for I2V."""

    def __init__(self, model_name="Wan2.1-Fun-V1.1-1.3B-InP", timestep_shift=5.0, is_causal=False,
                 local_attn_size=-1, sink_size=0, lora=False, lora_ckpt_path=None,
                 lora_alpha=1.0, upcast_dtype=None, hotload=False, lora_training=True):
        super().__init__()

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")

        if lora:
            self._setup_lora(model_name, lora_ckpt_path, lora_alpha, upcast_dtype, lora_training)

        self.model.eval()
        self.uniform_timestep = not is_causal
        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.seq_len = 32760
        self.post_init()

    def _setup_lora(self, model_name, lora_ckpt_path, lora_alpha, upcast_dtype, lora_training):
        print(f"Injecting LoRA into {model_name}")
        if lora_training:
            self.model = inject_adapter_in_model(gwtf_lora_config, self.model)

        if upcast_dtype is not None and upcast_dtype != "float32":
            upcast_dtype = type_dict[upcast_dtype]
            for param in self.model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)

        if lora_ckpt_path is not None:
            lora = load_state_dict_from_safetensors(lora_ckpt_path, torch_dtype=upcast_dtype, device=self.model.device)
            loader = GeneralLoRALoader(torch_dtype=upcast_dtype, device=self.model.device)
            loader.load(self.model, lora, alpha=lora_alpha)

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def forward(self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
                timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
                curr_y=None, crossattn_cache: Optional[List[dict]] = None,
                current_start: Optional[int] = None, clean_x: Optional[torch.Tensor] = None,
                aug_t: Optional[torch.Tensor] = None, cache_start: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        clip_fea = conditional_dict.get("clip_feature")
        if curr_y is not None:
            y = curr_y
        else:
            y = conditional_dict.get("y")
            if y is not None:
                y = y.permute(0, 2, 1, 3, 4)

        input_timestep = timestep[:, 0] if self.uniform_timestep else timestep

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len, kv_cache=kv_cache,
                crossattn_cache=crossattn_cache, current_start=current_start,
                cache_start=cache_start, clip_fea=clip_fea,
                y=y.permute(0, 2, 1, 3, 4) if y is not None else None
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len, clip_fea=clip_fea,
                y=y.permute(0, 2, 1, 3, 4) if y is not None else None
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        self.get_scheduler()
