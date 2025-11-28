from enum import Enum


class KernelType(str, Enum):
    RMSNORM = "rmsnorm"
    SWIGLU = "swiglu"
    FLASH_ATTENTION = "flash_attention"
    ROPE = "rope"
    MOE = "moe"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    NPU = "npu"
    XPU = "xpu"
