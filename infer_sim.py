"""I2V Flow inference with SDEdit from simulation results.

Based on infer_flow.py with SDEdit support added. Loads simulation data
(structured noise, frames, masks) from a simulation output directory and
uses CausalInferencePipelineSDEdit for simulation-guided video generation.

Example usage:
    python infer_sim.py \
        --checkpoint_path /path/to/model.pt \
        --sim_data_path result/tree/25-01_21-35-02/final_sim \
        --output_path ./output.mp4
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
import imageio

from vidgen import (
    WanImageEncoder,
    WanVideoVAE,
    WanVideoUnit_ImageEmbedderCLIP,
    WanVideoUnit_ImageEmbedderVAE,
    set_seed,
    apply_config_overrides,
    gpu,
    get_cuda_free_memory_gb,
    DynamicSwapInstaller,
    load_noise,
    load_first_frame,
    CausalInferencePipelineSDEdit
)

def load_sim_frames(frames_dir, height=480, width=832):
    """Load simulation frames from a directory of PNGs.

    Returns:
        Tensor of shape [1, C, T, H, W] normalized to [-1, 1].
    """
    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frame_*.png files found in {frames_dir}")

    frames = []
    for fp in frame_files:
        img = Image.open(fp).convert("RGB").resize((width, height))
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        frames.append(torch.from_numpy(arr))

    # [T, H, W, C] -> [C, T, H, W]
    frames_tensor = torch.stack(frames, dim=0).permute(3, 0, 1, 2).contiguous()
    return frames_tensor.unsqueeze(0)  # [1, C, T, H, W]


def load_sim_masks(mask_path, target_frames):
    """Load and temporally resize simulation masks.

    Args:
        mask_path: Path to a .pt mask file of shape [T_mask, H, W].
        target_frames: Target number of latent frames.

    Returns:
        Tensor of shape [1, target_frames, H, W] (bool).
    """
    masks = torch.load(mask_path, map_location="cpu", weights_only=True)  # [T_mask, H, W]
    T_mask = masks.shape[0]
    if T_mask != target_frames:
        # Nearest-neighbor temporal resize. Keep index tensor on the same device
        # as masks to avoid device mismatch when a global default device is set.
        indices = torch.linspace(
            0,
            T_mask - 1,
            steps=target_frames,
            device=masks.device,
        ).round().long().clamp(0, T_mask - 1)
        masks = masks[indices]
    return masks.unsqueeze(0)  # [1, T, H, W]


def _build_runtime_config(local_attn_size, denoising_step_list, mask_dropin_step, additional_args=None):
    default_config = {
        "independent_first_frame": False,
        "warp_denoising_step": True,
        "context_noise": 0,
        "causal": True,
        "i2v": True,
        "i2v_flow": True,
        "height": 480,
        "width": 832,
        "num_frame_per_block": 3,
        "denoising_step_list": denoising_step_list,
        "mask_dropin_step": mask_dropin_step,
        "model_kwargs": {
            "sink_size": 1,
            "local_attn_size": local_attn_size,
            "timestep_shift": 5.0,
        },
    }
    config = OmegaConf.create(default_config)
    if additional_args is not None:
        config = apply_config_overrides(config, additional_args)
    return config


class SimInferenceRunner:
    def __init__(self, checkpoint_path, use_ema=False, seed=42, local_attn_size=21, additional_args=None):
        self.checkpoint_path = checkpoint_path
        self.use_ema = use_ema
        self.seed = seed
        self.local_attn_size = local_attn_size
        self.additional_args = additional_args or []

        self.device = torch.device("cuda")
        set_seed(self.seed)
        torch.set_grad_enabled(False)

        print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
        self.low_memory = get_cuda_free_memory_gb(gpu) < 40

        init_config = _build_runtime_config(
            local_attn_size=self.local_attn_size,
            denoising_step_list=[800, 500, 250],
            mask_dropin_step=-1,
            additional_args=self.additional_args,
        )

        self.pipeline = CausalInferencePipelineSDEdit(init_config, device=self.device)
        self._load_checkpoint()

        self.pipeline = self.pipeline.to(dtype=torch.bfloat16)
        if self.low_memory:
            DynamicSwapInstaller.install_model(self.pipeline.text_encoder, device=gpu)
        else:
            self.pipeline.text_encoder.to(device=gpu)
        self.pipeline.generator.to(device=gpu)
        self.pipeline.vae.to(device=gpu)

        self.pipeline.processor_dtype = torch.float32
        self.pipeline.processor_device = gpu
        self.pipeline.processor_vae = WanVideoVAE().to(device=self.pipeline.processor_device, dtype=self.pipeline.processor_dtype)
        self.pipeline.processor_ienc = WanImageEncoder().to(device=self.pipeline.processor_device, dtype=self.pipeline.processor_dtype)

        self.pipeline.processor_vae.requires_grad_(False)
        self.pipeline.processor_ienc.requires_grad_(False)

        for p in self.pipeline.processor_vae.parameters():
            p.data = p.data.to(dtype=self.pipeline.processor_dtype)
        for b in self.pipeline.processor_vae.buffers():
            b.data = b.data.to(dtype=self.pipeline.processor_dtype)

        self.pipeline.processors = [
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
        ]

    def _load_checkpoint(self):
        if not self.checkpoint_path:
            return
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        key = 'generator_ema' if self.use_ema else 'generator'
        gen_state_dict = state_dict[key]
        try:
            self.pipeline.generator.load_state_dict(gen_state_dict)
        except Exception:
            gen_state_dict = {k.replace('._fsdp_wrapped_module', ''): v for k, v in gen_state_dict.items()}
            self.pipeline.generator.load_state_dict(gen_state_dict)

    def _update_case_params(self, denoising_step_list, mask_dropin_step):
        denoise_steps = torch.tensor(denoising_step_list, dtype=torch.long)
        if self.pipeline.args.warp_denoising_step:
            timesteps = torch.cat((self.pipeline.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32).cpu()))
            denoise_steps = timesteps[1000 - denoise_steps.cpu()]
        self.pipeline.denoising_step_list = denoise_steps

        first_step = self.pipeline.denoising_step_list[0]
        self.pipeline.sdedit = bool(first_step < len(self.pipeline.scheduler.timesteps))
        self.pipeline.mask_dropin_step = int(mask_dropin_step) if self.pipeline.sdedit else -1

    def infer_one(self, sim_data_path, output_path, eval_degradation=0.5):
        sim_data_path = Path(sim_data_path)
        sim_config_path = sim_data_path / "config.yaml"
        sim_config = OmegaConf.load(sim_config_path)
        denoising_step_list = list(sim_config.denoising_step_list)
        mask_dropin_step = int(sim_config.mask_dropin_step)
        num_output_frames = int(sim_config.num_output_frames)
        print(f"Loaded from config.yaml: denoising_step_list={denoising_step_list}, mask_dropin_step={mask_dropin_step}, num_output_frames={num_output_frames}")

        # We want the final video to be exactly `num_output_frames` pixel frames.
        # WanVideo generative blocks require latent frame count L to satisfy `L % 3 == 0`.
        # VAE decoding L latents produces `L * 4 - 3` pixel frames.
        # Find minimum valid L that generates >= num_output_frames pixel frames.
        min_pixels = num_output_frames
        target_latent_frames = 3
        while target_latent_frames * 4 - 3 < min_pixels:
            target_latent_frames += 3

        self._update_case_params(denoising_step_list, mask_dropin_step)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        noise_path = sim_data_path / "noises.npy"
        first_frame_path = sim_data_path / "resized_input_image.png"
        frames_dir = sim_data_path / "frames"

        print(f"Loading noise from: {noise_path}")
        noise_data = load_noise(
            noise_path=str(noise_path),
            target_frames=target_latent_frames,
            channel_dim=16,
            downsample_mode="nearest",
            eval_degradation=eval_degradation,
        )

        prompt_path = sim_data_path / "prompt.txt"
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()
        print(f"Prompt: {prompt}")

        print(f"Loading first frame from: {first_frame_path}")
        input_image = load_first_frame(str(first_frame_path), height=480, width=832)

        print(f"Loading simulation frames from: {frames_dir}")
        sim_frames = load_sim_frames(frames_dir, height=480, width=832)
        print(f"  Loaded {sim_frames.shape[2]} simulation frames")

        sim_masks = None
        if self.pipeline.mask_dropin_step > 0:
            mask_file = str(sim_data_path / "points_masks_downsampled.pt")
            if os.path.exists(mask_file):
                print(f"Loading object masks from: {mask_file}")
                sim_masks = load_sim_masks(mask_file, target_frames=target_latent_frames)
                sim_masks = sim_masks.to(device=self.device)
                print(f"  Object mask shape: {sim_masks.shape}")
            else:
                print(f"Warning: mask_dropin_step={self.pipeline.mask_dropin_step} but mask file not found: {mask_file}")
                print("  Proceeding without mask dropin.")

        sim_franka_masks = None
        franka_mask_file = sim_data_path / "mesh_masks_downsampled.pt"
        if franka_mask_file.exists():
            franka_masks_raw = load_sim_masks(str(franka_mask_file), target_frames=target_latent_frames)
            if franka_masks_raw.any():
                sim_franka_masks = franka_masks_raw.to(device=self.device)
                print(f"Loading franka masks from: {franka_mask_file}")
                print(f"  Franka mask shape: {sim_franka_masks.shape}")
            else:
                print("Franka masks are all False, skipping franka mask sdedit.")

        structured_noise = noise_data['structured_noise'].unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
        structured_noise_sde = noise_data.get('structured_noise_sde')
        if structured_noise_sde is not None:
            structured_noise_sde = structured_noise_sde.unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)

        sim_latent = None
        if self.pipeline.sdedit:
            print("Encoding simulation frames to latent space...")
            sim_frames_device = sim_frames.to(device=self.device, dtype=torch.bfloat16)
            sim_latent = self.pipeline.vae.encode_to_latent(sim_frames_device)
            sim_latent = sim_latent.to(device=self.device, dtype=torch.bfloat16)
            print(f"  sim_latent shape: {sim_latent.shape}")

            if sim_latent.shape[1] > target_latent_frames:
                sim_latent = sim_latent[:, :target_latent_frames]
            elif sim_latent.shape[1] < target_latent_frames:
                pad_size = target_latent_frames - sim_latent.shape[1]
                sim_latent = torch.cat([
                    sim_latent,
                    sim_latent[:, -1:].repeat(1, pad_size, 1, 1, 1),
                ], dim=1)
            print(f"  sim_latent shape (after align): {sim_latent.shape}")

        pixel_num_frames = target_latent_frames * 4 - 3
        batch = {
            'input_image': input_image.unsqueeze(0),
            'end_image': None,
            'height': 480,
            'width': 832,
            'num_frames': pixel_num_frames,
        }

        print("Generating video...")
        video, _ = self.pipeline.inference(
            noise=structured_noise,
            text_prompts=[prompt],
            return_latents=True,
            batch_sample=batch,
            sim_latent=sim_latent,
            sim_masks=sim_masks,
            sim_franka_masks=sim_franka_masks,
            low_memory=self.low_memory,
            device=self.device,
            structured_noise_sde=structured_noise_sde,
        )

        video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video = (255.0 * video[0]).to(torch.uint8)
        self.pipeline.vae.model.clear_cache()
        
        # truncate the video to the desired output frames
        video = video[:num_output_frames]

        print(f"Saving video to: {output_path}")
        imageio.mimwrite(output_path, video.numpy(), fps=24)
        print("Done!")

    def infer_from_csv(self, csv_path, result_root="result", eval_degradation=0.5):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"CSV is empty: {csv_path}")

        for idx, row in enumerate(rows, start=1):
            case_name = (row.get("case_name") or "").strip()
            output_tag = (row.get("sim_output_dir") or "000").strip()
            if not case_name:
                raise ValueError(f"Row {idx}: case_name is required")

            sim_data_path = row.get("sim_data_path")
            if not sim_data_path:
                sim_data_path = os.path.join(result_root, case_name, output_tag, "final_sim")

            output_path = row.get("infer_output_path")
            if not output_path:
                output_path = os.path.join(result_root, case_name, output_tag, "final.mp4")

            print(f"[{idx}/{len(rows)}] Inference for {case_name}")
            self.infer_one(
                sim_data_path=sim_data_path,
                output_path=output_path,
                eval_degradation=eval_degradation,
            )


def main():
    parser = argparse.ArgumentParser(description="I2V Flow Inference with SDEdit from Simulation")
    parser.add_argument("--csv_path", type=str, default="", help="CSV path for batch inference")
    parser.add_argument("--result_root", type=str, default="result", help="Default root folder for simulation/inference results")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--sim_data_path", type=str, default="",
                        help="Path to simulation final_sim directory (contains noises.npy, frames/, masks, etc.)")
    parser.add_argument("--output_path", type=str, default="./output_sim.mp4", help="Output video path")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_degradation", type=float, default=0.5,
                        help="Degradation level for noise (0.0 = pure structured noise)")
    parser.add_argument("--local_attn_size", type=int, default=21, help="Local attention size for causal model")

    args, additional_args = parser.parse_known_args()

    runner = SimInferenceRunner(
        checkpoint_path=args.checkpoint_path,
        use_ema=args.use_ema,
        seed=args.seed,
        local_attn_size=args.local_attn_size,
        additional_args=additional_args,
    )

    if args.csv_path:
        runner.infer_from_csv(
            csv_path=args.csv_path,
            result_root=args.result_root,
            eval_degradation=args.eval_degradation,
        )
        return

    if not args.sim_data_path:
        raise ValueError("Single-case mode requires --sim_data_path")

    runner.infer_one(
        sim_data_path=args.sim_data_path,
        output_path=args.output_path,
        eval_degradation=args.eval_degradation,
    )

if __name__ == "__main__":
    main()
