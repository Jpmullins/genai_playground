import random

import pytest
from datasets import load_dataset

from genaiplayground.v1.config.data_args import DataArguments
from genaiplayground.v1.core.data_engine import DataEngine


@pytest.mark.parametrize("num_samples", [16])
def test_map_dataset(num_samples: int):
    data_args = DataArguments(dataset="genaiplayground/v1-sft-demo")
    data_engine = DataEngine(data_args)
    original_data = load_dataset("genaiplayground/v1-sft-demo", split="train")
    indexes = random.choices(range(len(data_engine)), k=num_samples)
    for index in indexes:
        print(data_engine[index])
        assert data_engine[index] == {"_dataset_name": "default", **original_data[index]}


if __name__ == "__main__":
    test_map_dataset(1)
