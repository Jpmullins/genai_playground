from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataset."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "Cutoff length for the dataset."},
    )
