"""Causal inference pipeline with SDEdit support for simulation-guided video generation.

Based on vidgen/pipeline.py (CausalInferencePipeline) with SDEdit logic from
causal_inference_sdedit.py. This file is kept separate to avoid modifying the
cleaned vidgen module.
"""

from typing import List, Optional
import torch

from vidgen.models import WanDiffusionWrapper, WanI2VDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from vidgen.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from vidgen.utils import extract_subdim


class CausalInferencePipelineSDEdit(torch.nn.Module):
    """Pipeline for causal video generation inference with SDEdit support."""

    def __init__(self, args, device, generator=None, text_encoder=None, vae=None,
                 use_separate_encode_vae=False):
        super().__init__()

        if args.i2v_flow:
            self.i2v_flow = True
            Wrapper = WanI2VDiffusionWrapper
        else:
            self.i2v_flow = False
            Wrapper = WanDiffusionWrapper

        # Initialize models
        self.generator = Wrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Separate VAE for encoding sim_latents to avoid cache conflicts
        # with the decoder's streaming cache on self.vae
        if use_separate_encode_vae:
            self.encode_vae = WanVAEWrapper()
        else:
            self.encode_vae = self.vae

        # Initialize causal hyperparameters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)

        # Detect SDEdit mode: if the first denoising step < total scheduler steps,
        # we start from a noised version of sim_latent instead of pure noise.
        if self.denoising_step_list[0] < len(self.scheduler.timesteps):
            self.sdedit = True
        else:
            self.sdedit = False

        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        if self.sdedit:
            self.mask_dropin_step = getattr(args, "mask_dropin_step", -1)
            # Franka mask SDEdit: default to last denoising step (weak sdedit)
            self.franka_step = getattr(args, "franka_step", len(args.denoising_step_list) - 1)
        else:
            self.mask_dropin_step = -1
            self.franka_step = -1

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")
        print(f"SDEdit enabled: {self.sdedit}")
        if self.sdedit:
            print(f"  mask_dropin_step: {self.mask_dropin_step}")
            # print(f"  franka_step: {self.franka_step}")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        sim_latent: Optional[torch.Tensor] = None,
        sim_masks: Optional[torch.Tensor] = None,
        sim_franka_masks: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        batch_sample=None,
        profile: bool = False,
        low_memory: bool = False,
        skip_decoding: bool = False,
        structured_noise_sde: Optional[torch.Tensor] = None,
        device=None,
    ) -> torch.Tensor:
        """Perform causal video generation inference with optional SDEdit.

        Args:
            noise: Input noise [B, T, C, H, W].
            text_prompts: List of text prompts.
            initial_latent: Optional initial latent for I2V.
            sim_latent: Encoded simulation frames [B, T, C, H, W] for SDEdit.
            sim_masks: Object masks [B, T, H, W] (True = object region to keep generated).
            sim_franka_masks: Franka/mesh masks [B, T, H, W] (True = franka region, weak sdedit).
            return_latents: Whether to return latents alongside decoded video.
            batch_sample: Batch dict for processor conditioning.
            structured_noise_sde: Deterministic SDE noise from simulation.
            device: Device to use.
        """
        batch_size, num_frames, num_channels, height, width = noise.shape

        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        # Text encoding
        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        # Process additional conditioning (I2V flow processors)
        if hasattr(self, "processors"):
            for unit in self.processors:
                input_data = {"device": device, "torch_dtype": self.processor_dtype}
                for key in unit.input_params:
                    input_data[key] = batch_sample.get(key) if batch_sample else None
                for key in unit.onload_model_names:
                    if key == "image_encoder":
                        input_data["image_encoder"] = self.processor_ienc
                    if key == "vae":
                        input_data["vae"] = self.processor_vae
                unit_output = unit.process(**input_data)
                for k, v in unit_output.items():
                    conditional_dict[k] = v.to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros([batch_size, num_output_frames, num_channels, height, width],
                             device=noise.device, dtype=noise.dtype)

        # Profiling setup
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)

        ############################################################
        ########################## SDEdit ##########################
        bg_noise = None
        bg_noise_franka = None
        use_franka_sdedit = (
            self.franka_step >= 0
            and sim_franka_masks is not None
            and sim_franka_masks.any()
        )

        if self.sdedit:
            assert sim_latent is not None, "sim_latent is required for SDEdit mode"
            assert noise.shape == sim_latent.shape, (
                f"noise shape {noise.shape} != sim_latent shape {sim_latent.shape}"
            )

            # Add noise to simulated latent at the first denoising step
            sdedit_dropin_step = self.denoising_step_list[0]
            noise = self.scheduler.add_noise(
                sim_latent.flatten(0, 1),
                noise.flatten(0, 1),
                sdedit_dropin_step * torch.ones(
                    [batch_size * noise.size(1)], device=noise.device, dtype=torch.long
                )
            ).unflatten(0, noise.shape[:2])

            # Prepare background noise for mask dropin
            if self.mask_dropin_step > 0:
                mask_dropin_step_diffusion = self.denoising_step_list[self.mask_dropin_step]
                bg_noise = self.scheduler.add_noise(
                    sim_latent.flatten(0, 1),
                    noise.flatten(0, 1),
                    mask_dropin_step_diffusion * torch.ones(
                        [batch_size * noise.size(1)], device=noise.device, dtype=torch.long
                    )
                ).unflatten(0, noise.shape[:2])

            # Prepare franka mask background noise (weak sdedit at a late step)
            if use_franka_sdedit and self.franka_step < len(self.denoising_step_list):
                franka_step_diffusion = self.denoising_step_list[self.franka_step]
                bg_noise_franka = self.scheduler.add_noise(
                    sim_latent.flatten(0, 1),
                    noise.flatten(0, 1),
                    franka_step_diffusion * torch.ones(
                        [batch_size * noise.size(1)], device=noise.device, dtype=torch.long
                    )
                ).unflatten(0, noise.shape[:2])
        ############################################################

        # Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        curr_y = None
        full_y = conditional_dict["y"].permute(0, 2, 1, 3, 4) if "y" in conditional_dict else None

        # Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            if full_y is not None:
                curr_y = full_y[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Prepare mask/background for this block
            bg_noisy_input = None
            mask_current = None
            bg_noisy_input_franka = None
            mask_current_franka = None

            if self.sdedit and self.mask_dropin_step > 0 and bg_noise is not None:
                bg_noisy_input = bg_noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                if sim_masks is not None:
                    mask_current = sim_masks[
                        :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            if use_franka_sdedit and bg_noise_franka is not None:
                bg_noisy_input_franka = bg_noise_franka[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                mask_current_franka = sim_franka_masks[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep

                # Mask drop-in: replace background with sim-noised latent
                if (
                    self.mask_dropin_step > 0
                    and self.mask_dropin_step == index
                    and mask_current is not None
                    and bg_noisy_input is not None
                ):
                    # mask_current is True for object region (keep generated),
                    # False for background (replace with simulation)
                    noisy_input = torch.where(mask_current.unsqueeze(2), noisy_input, bg_noisy_input)

                # Franka mask drop-in: where franka mask is True, replace with
                # sim-noised latent (weak sdedit to keep structure close to simulation)
                if (
                    use_franka_sdedit
                    and self.franka_step == index
                    and mask_current_franka is not None
                    and bg_noisy_input_franka is not None
                ):
                    noisy_input = torch.where(mask_current_franka.unsqueeze(2), bg_noisy_input_franka, noisy_input)

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        curr_y=curr_y,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    if structured_noise_sde is not None:
                        sde_noise = extract_subdim(structured_noise_sde, 16, return_complement=False, channel_dim=2)
                        sde_noise = sde_noise[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                    else:
                        sde_noise = torch.randn_like(noisy_input)
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        sde_noise.flatten(0, 1),
                        next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        curr_y=curr_y,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Record output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            if full_y is not None:
                curr_y = full_y[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Update KV cache with clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                curr_y=curr_y,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_times.append(block_start.elapsed_time(block_end))

            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Decode output
        video = None
        if not skip_decoding:
            video = self.vae.decode_to_pixel(output, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time
            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        return (video, output) if return_latents else video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize KV cache for the WAN model."""
        kv_cache1 = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Initialize cross-attention cache for the WAN model."""
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
