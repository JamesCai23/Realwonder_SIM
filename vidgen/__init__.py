"""Minimal I2V Flow video generation module."""

from vidgen.pipeline import CausalInferencePipeline
from vidgen.pipeline_sdedit import CausalInferencePipelineSDEdit
from vidgen.models import (
    WanTextEncoder,
    WanImageEncoder,
    WanVAEWrapper,
    WanVideoVAE,
    WanDiffusionWrapper,
    WanI2VDiffusionWrapper,
)
from vidgen.utils import (
    set_seed,
    apply_config_overrides,
    TextImagePairDataset,
    WanVideoUnit_ImageEmbedderCLIP,
    WanVideoUnit_ImageEmbedderVAE,
    extract_subdim,
    preprocess_image,
    load_noise,
    load_prompt,
    load_first_frame,
)
from vidgen.memory import (
    gpu,
    cpu,
    get_cuda_free_memory_gb,
    DynamicSwapInstaller,
    move_model_to_device_with_memory_preservation,
)

__all__ = [
    "CausalInferencePipeline",
    "CausalInferencePipelineSDEdit",
    "WanTextEncoder",
    "WanImageEncoder",
    "WanVAEWrapper",
    "WanVideoVAE",
    "WanDiffusionWrapper",
    "WanI2VDiffusionWrapper",
    "set_seed",
    "apply_config_overrides",
    "TextImagePairDataset",
    "WanVideoUnit_ImageEmbedderCLIP",
    "WanVideoUnit_ImageEmbedderVAE",
    "extract_subdim",
    "preprocess_image",
    "load_noise",
    "load_prompt",
    "load_first_frame",
    "gpu",
    "cpu",
    "get_cuda_free_memory_gb",
    "DynamicSwapInstaller",
    "move_model_to_device_with_memory_preservation",
]
