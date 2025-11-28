import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np

import config

logger = logging.getLogger(__name__)

# Configure MLflow on import
_mlflow_conf = config.get_mlflow_config()
mlflow.set_tracking_uri(_mlflow_conf.tracking_uri)
mlflow.set_experiment(_mlflow_conf.experiment_name)
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", _mlflow_conf.s3_endpoint_url)
os.environ.setdefault("AWS_ACCESS_KEY_ID", config.AWS_ACCESS_KEY_ID)
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", config.AWS_SECRET_ACCESS_KEY)


def start_run(run_name: Optional[str] = None):
    return mlflow.start_run(run_name=run_name)


def log_params(params: Dict):
    mlflow.log_params(params)


def log_metrics(metrics: Dict, step: Optional[int] = None):
    mlflow.log_metrics(metrics, step=step)


def log_figure(fig, artifact_path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = Path(tmpdir) / "figure.png"
        fig.savefig(fig_path)
        mlflow.log_artifact(fig_path, artifact_path=artifact_path)


def log_numpy_array(name: str, array: np.ndarray, artifact_path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / f"{name}.npy"
        np.save(out_path, array)
        mlflow.log_artifact(out_path, artifact_path=artifact_path)
