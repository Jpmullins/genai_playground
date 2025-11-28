from typing import TYPE_CHECKING, Literal, TypedDict, Union

from typing_extensions import NotRequired


if TYPE_CHECKING:
    import datasets
    import torch
    import torch.utils.data
    import transformers

    Tensor = torch.Tensor
    TorchDataset = Union[torch.utils.data.Dataset, torch.utils.data.IterableDataset]
    HFDataset = Union[datasets.Dataset, datasets.IterableDataset]
    DataCollator = transformers.DataCollator
    DataLoader = torch.utils.data.DataLoader
    HFModel = transformers.PreTrainedModel
    DistModel = torch.nn.parallel.DistributedDataParallel
    Processor = Union[transformers.PreTrainedTokenizer, transformers.ProcessorMixin]
else:
    Tensor = None
    TorchDataset = None
    HFDataset = None
    DataCollator = None
    DataLoader = None
    HFModel = None
    DistModel = None
    Processor = None


class DatasetInfo(TypedDict, total=False):
    hf_hub_url: NotRequired[str]
    """HF hub dataset uri."""
    file_name: NotRequired[str]
    """Local file path."""
    dataset_dir: NotRequired[str]
    """Dataset directory, default to args.dataset_dir."""
    split: NotRequired[str]
    """Dataset split, default to "train"."""
    converter: NotRequired[str]
    """Dataset converter, default to None."""
    size: NotRequired[int]
    """Number of samples, default to all samples."""
    weight: NotRequired[float]
    """Dataset weight, default to 1.0."""
    streaming: NotRequired[bool]
    """Is streaming dataset, default to False."""


class Content(TypedDict):
    type: Literal["text", "tools", "reasoning", "tool_calls", "image_url"]
    value: str


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: list[Content]
    loss_weight: float


class SFTSample(TypedDict):
    messages: list[Message]
    extra_info: NotRequired[str]
    _dataset_name: NotRequired[str]


class DPOSample(TypedDict):
    chosen_messages: list[Message]
    rejected_messages: list[Message]
    extra_info: NotRequired[str]
    _dataset_name: NotRequired[str]


Sample = Union[SFTSample, DPOSample]


class Model(TypedDict):
    hf_model: HFModel
    """HF model."""
    dist_model: DistModel
    """Distributed model."""
