import logging
from typing import Callable, List, Dict, Optional

import config
from models import hf_model

try:
    from models import vllm_client
except Exception:  # vLLM optional
    vllm_client = None  # type: ignore

logger = logging.getLogger(__name__)


class ModelRouter:
    def __init__(
        self,
        hf_backend: Optional[Callable[[List[Dict[str, str]] | str], str]] = None,
        vllm_backend: Optional[Callable[[List[Dict[str, str]] | str], str]] = None,
    ) -> None:
        self.hf_backend = hf_backend
        self.vllm_backend = vllm_backend

    def generate_local_response(self, messages_or_text: List[Dict[str, str]] | str) -> str:
        if not self.vllm_backend:
            raise RuntimeError("vLLM backend not configured; ensure VLLM_SERVER_URL is reachable.")
        logger.info("Routing request to vLLM backend")
        return self.vllm_backend(messages_or_text)


_default_router: Optional[ModelRouter] = None


def get_default_router() -> ModelRouter:
    global _default_router
    if _default_router is None:
        vllm_fn = None
        if vllm_client is not None:
            vllm_fn = getattr(vllm_client, "generate_with_vllm", None)
        _default_router = ModelRouter(hf_backend=None, vllm_backend=vllm_fn)
    return _default_router
