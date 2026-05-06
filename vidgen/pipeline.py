"""Causal inference pipeline for video generation."""

from typing import List, Optional
import torch

from vidgen.models import WanDiffusionWrapper, WanI2VDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from vidgen.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from vidgen.utils import extract_subdim


class CausalInferencePipeline(torch.nn.Module):
    """Pipeline for causal video generation inference."""

    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
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

        # Initialize causal hyperparameters
        self.scheduler = self.generator.get_scheduler()
        # Keep denoising steps on CPU so they can safely index CPU scheduler timesteps.
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long, device="cpu")
        if args.warp_denoising_step:
            # Build both tensors from the same CPU source to avoid default-device side effects.
            timesteps_cpu = self.scheduler.timesteps.detach().to(device="cpu")
            zero_step = timesteps_cpu.new_tensor([0.0])
            timesteps = torch.cat((timesteps_cpu, zero_step))
            warp_indices = (1000 - self.denoising_step_list).to(device=timesteps.device, dtype=torch.long)
            self.denoising_step_list = timesteps[warp_indices]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(self, noise: torch.Tensor, text_prompts: List[str],
                  initial_latent: Optional[torch.Tensor] = None,
                  return_latents: bool = False, batch_sample=None,
                  profile: bool = False, low_memory: bool = False,
                  skip_decoding: bool = False,
                  structured_noise_sde: Optional[torch.Tensor] = None,
                  device=None) -> torch.Tensor:
        """Perform causal video generation inference."""
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

        # Process additional conditioning
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

            # Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep

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
