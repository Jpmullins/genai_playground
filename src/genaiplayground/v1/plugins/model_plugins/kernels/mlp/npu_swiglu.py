import re
import types

import torch

from .....extras.types import HFModel
from ....trainer_plugins.distributed.accelerate import is_torch_npu_available
from ..constants import DeviceType, KernelType
from ..registry import MetaSwiGluKernel


def _npu_swiglu_forward(self, hidden_state):
    import torch_npu

    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1)
    )


def _npu_swiglu_glm4_forward(self, hidden_states):
    import torch_npu

    up_states = self.gate_up_proj(hidden_states)
    gate, up_states = up_states.chunk(2, dim=-1)
    return self.down_proj(torch_npu.npu_swiglu(torch.cat((gate, up_states), dim=-1), dim=-1))


def _npu_swiglu_gemma3ntext_forward(self, hidden_states):
    import torch_npu

    gate_proj = self.gate_proj(hidden_states)
    if self.activation_sparsity > 0.0:
        gate_proj = self._gaussian_topk(gate_proj)
    down_proj = self.down_proj(
        torch_npu.npu_swiglu(torch.cat((gate_proj, self.up_proj(hidden_states)), dim=-1), dim=-1)
    )
    return down_proj


class NpuSwiGluKernel(MetaSwiGluKernel):
    type = KernelType.SWIGLU
    device = DeviceType.NPU
    kernel = _npu_swiglu_forward

    # Don't apply the kernel to the following modules
    expect_modules = frozenset(
        {
            "Qwen3VLMoeTextMLP",
            "Qwen3VLTextMLP",
            "Qwen3OmniMoeThinkerTextMLP",
            "Qwen3OmniMoeMLP",
            "Qwen3OmniMoeTalkerTextMLP",
            "Qwen3OmniMoeCode2WavMlp",
            "Qwen3NextMLP",
            "Qwen3MoeMLP",
            "Qwen3MLP",
            "Qwen2MLP",
            "Qwen2MoeMLP",
            "Qwen2_5_VLMLP",
            "Qwen2_5OmniMLP",
            "Llama4TextMLP",
            "LlamaMLP",
            "Glm4MLP",
            "Glm4MoeMLP",
            "Glm4vMoeTextMLP",
            "Gemma3MLP",
            "Gemma2MLP",
            "Gemma3nTextMLP",
            "Phi3MLP",
            "DeepseekV2MLP",
            "DeepseekV3MLP",
            "SeedOssMLP",
        }
    )

    @classmethod
    def apply(cls, model, **kwargs) -> "HFModel":
        if not is_torch_npu_available():
            return model

        # Mapping of specific mlp modules to their corresponding kernel implementations
        kernel_mapping = {
            "Glm4MLP": _npu_swiglu_glm4_forward,
            "Glm4vTextMLP": _npu_swiglu_glm4_forward,
            "Phi3MLP": _npu_swiglu_glm4_forward,
            "Gemma3nTextMLP": _npu_swiglu_gemma3ntext_forward,
        }

        swiglu_pattern = re.compile("MLP", re.IGNORECASE)
        for name, module in model.named_modules():
            # Match any module whose class name contains "MLP"
            if (
                re.search(swiglu_pattern, module.__class__.__name__)
                and module.__class__.__name__ in cls.expect_modules
            ):
                # Bind function as an instance method to preserve `self` semantics
                # and replace the original forward
                kernel_func = kernel_mapping.get(module.__class__.__name__, _npu_swiglu_forward)
                module.forward = types.MethodType(kernel_func, module)

        return model
