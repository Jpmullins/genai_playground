import logging
import shlex
import subprocess
import sys
import uuid
from typing import Dict, List

logger = logging.getLogger(__name__)


def _build_command(params: Dict[str, str | int | float | None]) -> List[str]:
    cmd = [sys.executable, "-m", "training.finetune"]
    mapping = {
        "model_name": "--model-name",
        "train_path": "--train-path",
        "eval_path": "--eval-path",
        "epochs": "--epochs",
        "batch_size": "--batch-size",
        "learning_rate": "--learning-rate",
        "output_dir": "--output-dir",
    }
    for key, flag in mapping.items():
        value = params.get(key)
        if value is None or value == "":
            continue
        cmd.extend([flag, str(value)])
    return cmd


def launch_finetune_subprocess(params: Dict[str, str | int | float | None]) -> Dict[str, str | int]:
    job_id = uuid.uuid4().hex
    cmd_list = _build_command(params)
    logger.info("Launching fine-tune subprocess: %s", " ".join(shlex.quote(c) for c in cmd_list))
    process = subprocess.Popen(cmd_list)
    return {
        "job_id": job_id,
        "pid": process.pid,
        "command": " ".join(shlex.quote(c) for c in cmd_list),
    }
