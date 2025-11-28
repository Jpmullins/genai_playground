from dataclasses import dataclass, field


@dataclass
class SampleArguments:
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum number of new tokens to generate."},
    )
