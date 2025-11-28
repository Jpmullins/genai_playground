import logging
from typing import Dict, List, Optional

from openai import OpenAI

import config

logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    return OpenAI()


def _default_model(model: Optional[str]) -> str:
    return model or config.OPENAI_MODEL_NAME


def chat_via_responses(messages: List[Dict[str, str]] | str, model: Optional[str] = None) -> str:
    client = get_client()
    model_name = _default_model(model)
    logger.info("Calling OpenAI Responses API with model=%s", model_name)
    response = client.responses.create(
        model=model_name,
        input=messages,
        max_output_tokens=256,
        temperature=0.6,
    )
    return response.output_text


def chat_via_chat_completions(messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
    client = get_client()
    model_name = _default_model(model)
    logger.info("Calling OpenAI Chat Completions with model=%s", model_name)
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=256,
        temperature=0.6,
    )
    return completion.choices[0].message.content or ""


def chat_completion(messages: List[Dict[str, str]], use_responses: Optional[bool] = None, model: Optional[str] = None) -> str:
    if use_responses is None:
        use_responses = config.USE_RESPONSES_API
    if use_responses:
        return chat_via_responses(messages=messages, model=model)
    return chat_via_chat_completions(messages=messages, model=model)
