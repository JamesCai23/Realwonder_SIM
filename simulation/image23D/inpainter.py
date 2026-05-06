import torch
from diffusers.utils import load_image, check_min_version
from submodules.flux_controlnet_inpainting.controlnet_flux import FluxControlNetModel
from submodules.flux_controlnet_inpainting.transformer_flux import FluxTransformer2DModel
from submodules.flux_controlnet_inpainting.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
from simulation.utils import dilate_binary_mask, smooth_segmentation_mask_255
import sys
import os
sys.path.append(os.path.abspath("submodules/flux_controlnet_inpainting"))

check_min_version("0.30.2")


def _resolve_torch_dtype_from_env(default_dtype: torch.dtype) -> torch.dtype:
    raw = os.environ.get("REALWONDER_INPAINT_DTYPE", "").strip().lower()
    if not raw:
        return default_dtype
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"fp32", "float32", "full"}:
        return torch.float32
    return default_dtype


class FluxInpainter:
    _PIPE_CACHE = {}

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = _resolve_torch_dtype_from_env(torch_dtype)
        self.offload = os.environ.get("REALWONDER_INPAINT_OFFLOAD", "none").strip().lower()
        self.allow_fp16_fallback = os.environ.get("REALWONDER_INPAINT_FP16_FALLBACK", "1").strip() != "0"
        self.pipe = None
        self.load_model()
        
    def load_model(self):
        """Load the FLUX ControlNet inpainting model and pipeline"""
        cache_key = (str(self.device), str(self.torch_dtype), str(self.offload))
        cached_pipe = self._PIPE_CACHE.get(cache_key)
        if cached_pipe is not None:
            self.pipe = cached_pipe
            print(f"[FluxInpainter] Reused cached pipeline: key={cache_key}")
            return

        # Load ControlNet
        controlnet = FluxControlNetModel.from_pretrained(
            "/home/lff/data1/cym/physical_data/RealWonder/checkpoints/FLUX.1-dev-Controlnet-Inpainting-Beta", 
            torch_dtype=self.torch_dtype
        )
        
        # Load Transformer
        transformer = FluxTransformer2DModel.from_pretrained(
            "/home/lff/data1/cym/physical_data/RealWonder/checkpoints/FLUX.1-dev",
            subfolder='transformer',
            torch_dtype=self.torch_dtype
        )
        
        # Build pipeline
        self.pipe = FluxControlNetInpaintingPipeline.from_pretrained(
            "/home/lff/data1/cym/physical_data/RealWonder/checkpoints/FLUX.1-dev",
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=self.torch_dtype
        )

        if self.device.startswith("cuda"):
            try:
                if self.offload == "sequential":
                    self.pipe.enable_sequential_cpu_offload()
                elif self.offload == "cpu":
                    self.pipe.enable_model_cpu_offload()
                else:
                    self.pipe = self.pipe.to(self.device)
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                print("[FluxInpainter] CUDA OOM during pipeline load; switching to sequential CPU offload")
                torch.cuda.empty_cache()
                self.pipe.enable_sequential_cpu_offload()
                self.offload = "sequential"
        else:
            self.pipe = self.pipe.to(self.device)

        # Keep dtype on key modules; avoid forcing full relocation when offloading.
        try:
            self.pipe.transformer.to(dtype=self.torch_dtype)
            self.pipe.controlnet.to(dtype=self.torch_dtype)
        except Exception:
            pass

        self._PIPE_CACHE[cache_key] = self.pipe
        print(f"[FluxInpainter] Cached pipeline: key={cache_key}")
        
        print("Model loaded successfully")

    @classmethod
    def clear_cache(cls, device: str | None = None):
        target_device = None if device is None else str(device)
        keys_to_remove = []

        for cache_key in list(cls._PIPE_CACHE.keys()):
            key_device = str(cache_key[0]) if isinstance(cache_key, tuple) and len(cache_key) > 0 else None
            if target_device is None or key_device == target_device:
                keys_to_remove.append(cache_key)

        for cache_key in keys_to_remove:
            pipe = cls._PIPE_CACHE.pop(cache_key, None)
            if pipe is not None:
                del pipe

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    @staticmethod
    def _image_stats(image: Image.Image) -> tuple[float, float]:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        if arr.size == 0:
            return 0.0, 0.0
        return float(arr.max()), float(arr.mean())

    @staticmethod
    def _is_near_black_pil(image: Image.Image, max_thr: float = 0.03, mean_thr: float = 0.01) -> bool:
        max_v, mean_v = FluxInpainter._image_stats(image)
        return max_v <= max_thr and mean_v <= mean_thr

    def _reload_with_dtype(self, new_dtype: torch.dtype) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        self.torch_dtype = new_dtype
        self.load_model()

    def _run_generation_with_retries(
        self,
        *,
        prompt: str,
        req_h: int,
        req_w: int,
        masked_image_pil: Image.Image,
        mask: Image.Image,
        num_inference_steps: int,
        seed: int,
        generator_device: str,
        controlnet_conditioning_scale: float,
        guidance_scale: float,
        negative_prompt: str,
        true_guidance_scale: float,
    ) -> tuple[Image.Image | None, list[tuple[int, float, float]]]:
        retry_times = int(os.environ.get("REALWONDER_INPAINT_RETRY", "3"))
        retry_times = max(1, retry_times)
        retry_stats: list[tuple[int, float, float]] = []
        result = None

        for retry_idx in range(retry_times):
            curr_seed = int(seed + retry_idx)
            curr_generator = torch.Generator(device=generator_device).manual_seed(curr_seed)

            try:
                result = self.pipe(
                    prompt=prompt,
                    height=req_h,
                    width=req_w,
                    control_image=masked_image_pil,
                    control_mask=mask,
                    num_inference_steps=num_inference_steps,
                    generator=curr_generator,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    true_guidance_scale=true_guidance_scale
                ).images[0]

                # Force async CUDA failures to surface at the true inpaint callsite.
                if generator_device.startswith("cuda"):
                    torch.cuda.synchronize(generator_device)
            except RuntimeError as e:
                err = str(e)
                err_l = err.lower()
                if "device-side assert triggered" in err_l or "indexselectlargeindex" in err_l or "srcselectdimsize" in err_l:
                    raise RuntimeError(
                        "Flux inpainting hit a CUDA device-side assert (likely invalid token index or a prior async CUDA fault). "
                        "The pipeline now validates T5 token ids before encoder forward; please re-run this case with the latest code. "
                        f"Original error: {err}"
                    ) from e
                raise

            max_v, mean_v = self._image_stats(result)
            retry_stats.append((curr_seed, max_v, mean_v))

            if not self._is_near_black_pil(result):
                break

            print(
                f"[FluxInpainter] near-black output at retry {retry_idx + 1}/{retry_times} "
                f"(seed={curr_seed}, max={max_v:.5f}, mean={mean_v:.5f}), retrying",
                flush=True,
            )

        return result, retry_stats
        
    def __call__(self, image, mask, prompt="", size=(512, 512), 
                      num_inference_steps=24, controlnet_conditioning_scale=0.9,
                      guidance_scale=3.5, negative_prompt="", true_guidance_scale=3.5,
                      seed=42):
        """Run inpainting with the given parameters"""
        if self.pipe is None:
            raise ValueError("Model not loaded. Please call load_model() first.")

        generator_device = self.device if self.device.startswith("cuda") else "cpu"

        req_w, req_h = int(size[0]), int(size[1])

        mask = dilate_binary_mask(mask, size=(req_w, req_h), kernel_size=50, iterations=1)
        mask = smooth_segmentation_mask_255(mask, blur_kernel_size=51, blur_sigma=5.0, threshold=60, binary_output=True, morph_close=True, morph_kernel_size=7, return_pil=True)

        image_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        image_np = np.array(Image.fromarray(image_np).resize((req_w, req_h), resample=Image.BILINEAR))
        mask_np = np.array(mask)

        mask_3c = np.repeat(mask_np[:, :, None] == 0, 3, axis=2)
        masked_image = np.where(mask_3c, image_np, 255)

        masked_image_pil = Image.fromarray(masked_image)
        # masked_image_pil.save('debug/masked_image.png')

        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            prompt = "empty background, completely remove the foreground object, seamless padding, high quality, highly detailed, photorealistic"
        if not isinstance(negative_prompt, str) or len(negative_prompt.strip()) == 0:
            negative_prompt = "foreground object, distorted, artifacts, blurry, low quality, bad anatomy, deformed, ugly"

        result, retry_stats = self._run_generation_with_retries(
            prompt=prompt,
            req_h=req_h,
            req_w=req_w,
            masked_image_pil=masked_image_pil,
            mask=mask,
            num_inference_steps=num_inference_steps,
            seed=seed,
            generator_device=generator_device,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            true_guidance_scale=true_guidance_scale,
        )

        if (result is None or self._is_near_black_pil(result)) and self.allow_fp16_fallback and self.torch_dtype == torch.bfloat16:
            print(
                "[FluxInpainter] all bf16 attempts were near-black; reloading pipeline in fp16 and retrying",
                flush=True,
            )
            self._reload_with_dtype(torch.float16)
            result, retry_stats = self._run_generation_with_retries(
                prompt=prompt,
                req_h=req_h,
                req_w=req_w,
                masked_image_pil=masked_image_pil,
                mask=mask,
                num_inference_steps=num_inference_steps,
                seed=seed,
                generator_device=generator_device,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                true_guidance_scale=true_guidance_scale,
            )

        if result is None or self._is_near_black_pil(result):
            stats_str = ", ".join(
                [f"seed={s},max={mx:.5f},mean={mn:.5f}" for s, mx, mn in retry_stats]
            )
            raise RuntimeError(
                "Flux inpainting produced near-black outputs. "
                f"size={req_w}x{req_h}, device={self.device}, dtype={self.torch_dtype}, "
                f"offload={self.offload}, attempts=[{stats_str}]"
            )

        return result
