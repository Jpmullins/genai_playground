import unittest
from unittest.mock import MagicMock, patch

from transformers import AutoModelForCausalLM


class TestKernelPlugin(unittest.TestCase):
    @patch("torch.accelerator.current_accelerator")
    def test_apply_kernel(self, mock_get_accelerator):
        mock_device = MagicMock()
        mock_device.type = "npu"
        mock_get_accelerator.return_value = mock_device

        model = AutoModelForCausalLM.from_pretrained("genaiplayground/tiny-random-qwen2.5")

        original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward
        original_swiglu_forward = model.model.layers[0].mlp.forward

        from genaiplayground.v1.plugins.model_plugins.kernels.mlp import npu_swiglu
        from genaiplayground.v1.plugins.model_plugins.kernels.registry import apply_kernel
        from genaiplayground.v1.plugins.model_plugins.kernels.rms_norm import npu_rms_norm
        from genaiplayground.v1.plugins.model_plugins.kernels.rope import npu_rope

        apply_kernel(model, npu_rope.NpuRoPEKernel)

        model = apply_kernel(model, npu_rms_norm.NpuRMSNormKernel)
        assert model.model.layers[0].input_layernorm is not original_rmsnorm_forward

        model = apply_kernel(model, npu_swiglu.NpuSwiGluKernel)
        assert model.model.layers[0].mlp.forward is not original_swiglu_forward


class Test_Use_V1_Kernels(unittest.TestCase):
    @patch("torch.accelerator.current_accelerator")
    def test_use_v1_kernels(self, mock_get_accelerator):
        mock_device = MagicMock()
        mock_device.type = "npu"
        mock_get_accelerator.return_value = mock_device

        model = AutoModelForCausalLM.from_pretrained("genaiplayground/tiny-random-qwen2.5")

        original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward
        original_swiglu_forward = model.model.layers[0].mlp.forward

        from genaiplayground.v1.plugins.model_plugins.kernels.registry import apply_available_kernels

        model = apply_available_kernels(model)

        assert model.model.layers[0].input_layernorm is not original_rmsnorm_forward
        assert model.model.layers[0].mlp.forward is not original_swiglu_forward
