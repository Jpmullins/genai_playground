import os
from dataclasses import dataclass

import torch
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    s3_endpoint_url: str


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Core settings
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
HF_MODEL_NAME: str = os.getenv("HF_MODEL_NAME", "gpt2")
OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
USE_RESPONSES_API: bool = _get_env_bool("USE_RESPONSES_API", True)
USE_VLLM: bool = True  # vLLM is now the default path
VLLM_SERVER_URL: str = os.getenv("VLLM_SERVER_URL", "http://vllm:8000/v1")
VLLM_MODEL_NAME: str | None = os.getenv("VLLM_MODEL_NAME")

# MLflow
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "llm_pipeline")
MLFLOW_RUN_NAME: str = os.getenv("MLFLOW_RUN_NAME", "dev_run")
MLFLOW_S3_ENDPOINT_URL: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")


def get_device() -> torch.device:
    """Return CUDA device if available else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_mlflow_config() -> MLflowConfig:
    return MLflowConfig(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        s3_endpoint_url=MLFLOW_S3_ENDPOINT_URL,
    )


def use_vllm() -> bool:
    """Return True only when vLLM is requested and CUDA is available."""
    return USE_VLLM and torch.cuda.is_available()
