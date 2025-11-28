import logging
from functools import lru_cache
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

import config

logger = logging.getLogger(__name__)


_LAST_MODEL_NAME: str | None = None


@lru_cache(maxsize=2)
def load_model(model_name: str | None = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer, torch.device]:
    resolved_name = model_name or config.HF_MODEL_NAME
    device = config.get_device()
    logger.info("Loading HF model %s on %s", resolved_name, device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_name, token=config.HUGGINGFACE_HUB_TOKEN if hasattr(config, "HUGGINGFACE_HUB_TOKEN") else None)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(resolved_name, token=config.HUGGINGFACE_HUB_TOKEN if hasattr(config, "HUGGINGFACE_HUB_TOKEN") else None)
        active_name = resolved_name
    except Exception as exc:
        fallback = "distilgpt2"
        if resolved_name != fallback:
            logger.warning("Failed to load %s (%s); falling back to %s", resolved_name, exc, fallback)
            model, tokenizer, device = load_model(fallback)
            active_name = fallback
        else:
            logger.error("Failed to load fallback model %s: %s", fallback, exc)
            raise
    else:
        model.to(device)
        model.eval()
    global _LAST_MODEL_NAME
    _LAST_MODEL_NAME = active_name
    return model, tokenizer, device


def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts) + "\nassistant:"


def generate_response(messages_or_text: List[Dict[str, str]] | str, max_new_tokens: int = 128) -> str:
    model, tokenizer, device = load_model()

    if isinstance(messages_or_text, list):
        prompt_text = _messages_to_text(messages_or_text)
    else:
        prompt_text = messages_or_text

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    # Prefer content after the last assistant marker, else trim the prompt prefix
    if "assistant:" in text:
        candidate = text.split("assistant:")[-1].strip()
        if candidate:
            return candidate
    if text.startswith(prompt_text):
        return text[len(prompt_text) :].strip()
    return text.strip()


def get_active_model_name() -> str:
    return _LAST_MODEL_NAME or config.HF_MODEL_NAME
