import logging
from typing import Dict, List

try:
    import requests
except ImportError as exc:  # pragma: no cover - optional dep
    requests = None  # type: ignore
    _import_error = exc
else:
    _import_error = None

import config

logger = logging.getLogger(__name__)


class VLLMNotAvailable(Exception):
    pass


def _ensure_requests_available():
    if requests is None:
        raise VLLMNotAvailable(f"requests not available: {_import_error}")


def _normalize_messages(messages_or_text: List[Dict[str, str]] | str) -> List[Dict[str, str]]:
    if isinstance(messages_or_text, list):
        return messages_or_text
    return [{"role": "user", "content": str(messages_or_text)}]


def generate_with_vllm(messages_or_text: List[Dict[str, str]] | str) -> str:
    """Call vLLM OpenAI-compatible /chat/completions endpoint."""
    _ensure_requests_available()
    if not config.use_vllm():
        raise VLLMNotAvailable("vLLM disabled or CUDA unavailable")

    url = f"{config.VLLM_SERVER_URL}/chat/completions"
    payload = {
        "model": config.VLLM_MODEL_NAME or "",
        "messages": _normalize_messages(messages_or_text),
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
    except Exception as exc:  # broad but we want graceful fallback
        logger.error("vLLM request failed: %s", exc)
        raise VLLMNotAvailable(str(exc)) from exc

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Unexpected vLLM response schema: %s", exc)
        raise VLLMNotAvailable("Invalid vLLM response schema") from exc
