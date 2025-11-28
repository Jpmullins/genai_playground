from ..config.sample_args import SampleArguments


class ChatSampler:
    def __init__(self, sample_args: SampleArguments) -> None:
        self.args = sample_args
