import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from monitoring import mlflow_utils

logger = logging.getLogger(__name__)


def register_activation_hooks(model: PreTrainedModel, layer_names: List[str]) -> Dict[str, torch.Tensor]:
    activations: Dict[str, torch.Tensor] = {}

    def get_hook(name: str):
        def hook(_module, _input, output):
            activations[name] = output.detach().cpu()

        return hook

    named_modules = dict(model.named_modules())
    for name in layer_names:
        module = named_modules.get(name)
        if module is None:
            logger.warning("Layer %s not found; skipping hook", name)
            continue
        module.register_forward_hook(get_hook(name))
    return activations


def compute_activations(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, text: str, layer_names: List[str]
) -> Dict[str, np.ndarray]:
    activations = register_activation_hooks(model, layer_names)
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        _ = model(**inputs)
    return {name: tensor.numpy() for name, tensor in activations.items()}


def plot_activation_summary(activations: Dict[str, np.ndarray]):
    figures: Dict[str, plt.Figure] = {}
    for name, act in activations.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        flat = act.flatten()
        axes[0].hist(flat, bins=50, color="steelblue")
        axes[0].set_title(f"Histogram: {name}")
        axes[1].imshow(act.squeeze(), aspect="auto", interpolation="nearest")
        axes[1].set_title(f"Heatmap: {name}")
        plt.tight_layout()
        figures[name] = fig
    return figures


def log_activations_to_mlflow(activations: Dict[str, np.ndarray], prefix: str = "activations") -> None:
    for name, array in activations.items():
        artifact_path = f"{prefix}/{name}"
        mlflow_utils.log_numpy_array(name, array, artifact_path=artifact_path)
    figs = plot_activation_summary(activations)
    for name, fig in figs.items():
        mlflow_utils.log_figure(fig, artifact_path=f"{prefix}/{name}")
