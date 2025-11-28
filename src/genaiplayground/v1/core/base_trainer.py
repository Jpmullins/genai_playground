from typing import Any

from ..config.training_args import TrainingArguments
from ..extras.types import Model, Processor, Tensor, TorchDataset


class DataCollator:
    """Default Data collator."""

    def __init__(self, processor: Processor) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        """Collate features into a batch."""
        for feature in features:
            pass

        # sft: messages
        # dpo: chosen_messages, rejected_messages


class BaseTrainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: Model,
        processor: Processor,
        dataset: TorchDataset,
        data_collator: DataCollator,
    ) -> None:
        self.args = args
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.data_collator = data_collator
        self.optimizer = None
        self.lr_scheduler = None

    def create_dataloader(self) -> None:
        pass

    def fit(self) -> None:
        pass
