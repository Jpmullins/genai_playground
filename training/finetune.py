import argparse
import logging
from pathlib import Path
from typing import Optional

import mlflow
import torch
from transformers import Trainer, TrainingArguments

import config
from models import hf_model
from monitoring import mlflow_utils
from training.dataset import load_supervised_dataset

logger = logging.getLogger(__name__)


def finetune_hf_model(
    model_name: Optional[str],
    train_path: str,
    eval_path: Optional[str],
    output_dir: str,
    num_train_epochs: float,
    batch_size: int,
    learning_rate: float,
    logging_steps: int = 10,
) -> None:
    if model_name:
        config.HF_MODEL_NAME = model_name
    model, tokenizer, device = hf_model.load_model(model_name)
    logger.info("Using device %s for fine-tuning", device)

    train_dataset = load_supervised_dataset(train_path, tokenizer)
    eval_dataset = load_supervised_dataset(eval_path, tokenizer) if eval_path else None

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="epoch",
        report_to=["none"],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # simple perplexity proxy
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = torch.tensor(logits[..., :-1, :])
        shift_labels = torch.tensor(labels[..., 1:])
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        return {"eval_loss": loss.item()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_dataset else None,
    )

    params = {
        "model_name": config.HF_MODEL_NAME,
        "train_path": train_path,
        "eval_path": eval_path or "",
        "epochs": num_train_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }

    with mlflow_utils.start_run(run_name=config.MLFLOW_RUN_NAME) as run:
        mlflow_utils.log_params(params)
        logger.info("Starting training run %s", run.info.run_id)
        train_result = trainer.train()
        mlflow_utils.log_metrics(train_result.metrics)

        if eval_dataset:
            eval_metrics = trainer.evaluate()
            mlflow_utils.log_metrics(eval_metrics)

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        mlflow.log_artifacts(output_dir, artifact_path="model")
        logger.info("Artifacts logged to MLflow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a HF causal LM with MLflow logging")
    parser.add_argument("--model-name", type=str, default=None, help="Base HF model name")
    parser.add_argument("--train-path", type=str, required=True, help="Path to JSONL training data")
    parser.add_argument("--eval-path", type=str, default=None, help="Optional eval JSONL path")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Where to store checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    finetune_hf_model(
        model_name=args.model_name,
        train_path=args.train_path,
        eval_path=args.eval_path,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
