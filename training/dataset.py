import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SupervisedDataset(Dataset):
    def __init__(self, encodings: List[Dict[str, torch.Tensor]]):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.encodings[idx]
        return {k: torch.tensor(v) for k, v in item.items()}


def load_supervised_dataset(path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512) -> Dataset:
    records: List[Dict[str, Any]] = []
    data_path = Path(path)
    with data_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)

    encodings: List[Dict[str, Any]] = []
    for rec in records:
        instruction = rec.get("instruction", "").strip()
        output = rec.get("output", "").strip()
        text = f"Instruction: {instruction}\nResponse: {output}"
        tokenized = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenized["input_ids"].copy()
        # Mask padding tokens from loss
        labels = [token if mask == 1 else -100 for token, mask in zip(labels, tokenized["attention_mask"])]
        tokenized["labels"] = labels
        encodings.append(tokenized)

    return SupervisedDataset(encodings)
