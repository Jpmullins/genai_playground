import logging

import mlflow

from monitoring import mlflow_utils

logger = logging.getLogger(__name__)

_tracing_enabled = False


def setup_tracing() -> None:
    global _tracing_enabled
    if _tracing_enabled:
        return
    # Ensure MLflow is configured
    _ = mlflow_utils
    mlflow.openai.autolog()
    logger.info("Enabled MLflow OpenAI autologging")
    _tracing_enabled = True


def demo_traced_openai_call() -> str:
    from openai import OpenAI

    setup_tracing()
    client = OpenAI()
    with mlflow_utils.start_run(run_name="demo-openai-call"):
        resp = client.responses.create(
            model="gpt-4o-mini",
            input="Say hello from the demo run.",
            max_output_tokens=16,
        )
        text = resp.output_text
    return text
