"""GPU memory management utilities."""

import torch
import copy
from contextlib import contextmanager

# Device references
cpu = torch.device('cpu')
gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
gpu_complete_modules = []


class DynamicSwapInstaller:
    """Dynamic model swapping for memory-efficient inference."""

    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)


def get_cuda_free_memory_gb(device=None):
    """Get free CUDA memory in GB."""
    if device is None:
        device = gpu
    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats['active_bytes.all.current']
    bytes_reserved = memory_stats['reserved_bytes.all.current']
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024 ** 3)


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    """Move model to device while preserving minimum free memory."""
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')
    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) <= preserved_memory_gb:
            torch.cuda.empty_cache()
            return
        if hasattr(m, 'weight'):
            m.to(device=target_device)
    model.to(device=target_device)
    torch.cuda.empty_cache()


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    """Offload model from device to preserve memory."""
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')
    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) >= preserved_memory_gb:
            torch.cuda.empty_cache()
            return
        if hasattr(m, 'weight'):
            m.to(device=cpu)
    model.to(device=cpu)
    torch.cuda.empty_cache()


@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers: bool = False):
    """Context manager for initializing weights on a specific device."""
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)
        return wrapper

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def cast_to(weight, dtype, device):
    """Cast weight to specified dtype and device."""
    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight)
    return r


class AutoWrappedLinear(torch.nn.Linear):
    """Auto-wrapped linear layer with VRAM management and LoRA support."""

    def __init__(self, module: torch.nn.Linear, offload_dtype, offload_device, onload_dtype,
                 onload_device, computation_dtype, computation_device, vram_limit, name="", **kwargs):
        with init_weights_on_device(device=torch.device("meta")):
            super().__init__(in_features=module.in_features, out_features=module.out_features,
                           bias=module.bias is not None, dtype=offload_dtype, device=offload_device)
        self.weight = module.weight
        self.bias = module.bias
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.vram_limit = vram_limit
        self.state = 0
        self.name = name
        self.lora_A_weights = []
        self.lora_B_weights = []
        self.lora_merger = None
        self.enable_fp8 = computation_dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]

    def check_free_vram(self):
        gpu_mem_state = torch.cuda.mem_get_info(self.computation_device)
        used_memory = (gpu_mem_state[1] - gpu_mem_state[0]) / (1024 ** 3)
        return used_memory < self.vram_limit

    def keep(self):
        if self.state != 2:
            self.to(dtype=self.computation_dtype, device=self.computation_device)
            self.state = 2

    def forward(self, x, *args, **kwargs):
        if self.state == 2:
            weight, bias = self.weight, self.bias
        else:
            if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
                weight, bias = self.weight, self.bias
            elif self.vram_limit is not None and self.check_free_vram():
                self.keep()
                weight, bias = self.weight, self.bias
            else:
                weight = cast_to(self.weight, self.computation_dtype, self.computation_device)
                bias = None if self.bias is None else cast_to(self.bias, self.computation_dtype, self.computation_device)

        out = torch.nn.functional.linear(x, weight, bias)

        if len(self.lora_A_weights) == 0:
            return out
        elif self.lora_merger is None:
            for lora_A, lora_B in zip(self.lora_A_weights, self.lora_B_weights):
                out = out + x @ lora_A.T @ lora_B.T
        else:
            lora_output = []
            for lora_A, lora_B in zip(self.lora_A_weights, self.lora_B_weights):
                lora_output.append(x @ lora_A.T @ lora_B.T)
            lora_output = torch.stack(lora_output)
            out = self.lora_merger(out, lora_output)
        return out
