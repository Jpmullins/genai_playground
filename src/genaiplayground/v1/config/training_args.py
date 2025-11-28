from dataclasses import dataclass, field


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default="",
        metadata={"help": "Path to the output directory."},
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size for training."},
    )
    global_batch_size: int = field(
        default=1,
        metadata={"help": "Global batch size for training."},
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for training."},
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Use bf16 for training."},
    )
