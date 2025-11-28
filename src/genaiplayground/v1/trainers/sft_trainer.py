from ..config.parser import get_args
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine


class SFTTrainer(BaseTrainer):
    pass


def run_sft():
    model_args, data_args, training_args, _ = get_args()
    model_engine = ModelEngine(model_args)
    data_engine = DataEngine(data_args)
    model = model_engine.get_model()
    processor = model_engine.get_processor()
    data_loader = data_engine.get_data_loader(processor)
    trainer = SFTTrainer(training_args, model, processor, data_loader)
    trainer.fit()
