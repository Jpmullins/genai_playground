from ..config.model_args import ModelArguments
from ..extras.types import Model, Processor


class ModelEngine:
    def __init__(self, model_args: ModelArguments) -> None:
        self.args = model_args

    def get_model(self) -> Model:
        pass

    def get_processor(self) -> Processor:
        pass
