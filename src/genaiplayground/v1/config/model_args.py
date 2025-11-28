from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model: str = field(
        metadata={"help": "Path to the model or model identifier from Hugging Face."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code from Hugging Face."},
    )
