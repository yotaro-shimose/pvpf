from typing import List

import tensorflow.keras as keras
from ray import tune

from pvpf.model.convlstm import ConvLSTMBlockParam, ConvLSTMRegressor
from pvpf.token.training_token import TRAINING_TOKENS
from pvpf.trainer.trainer import ModelParam, TrainingConfig, tune_trainer
from pvpf.validation.validate_analysis import validate_analysis


class ConvLSTMRegressorParam(ModelParam):
    block_params: List[ConvLSTMBlockParam]


def model_builder(param: ModelParam) -> keras.Model:
    return ConvLSTMRegressor(**param)


if __name__ == "__main__":
    token = TRAINING_TOKENS["masked_small"]
    model_param = ConvLSTMRegressorParam(
        block_params=[
            ConvLSTMBlockParam(num_filters=16, kernel_size=5, pooling=None),
            ConvLSTMBlockParam(num_filters=64, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=128, kernel_size=3, pooling=2),
            ConvLSTMBlockParam(num_filters=256, kernel_size=2, pooling=None),
        ]
    )
    config = TrainingConfig(
        model_builder=model_builder,
        model_param=model_param,
        batch_size=128,
        num_epochs=50,
        learning_rate=tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
        training_property=token,
        shuffle_buffer=500,  # carefully set this value to avoid OOM
    )
    resources_per_trial = {"cpu": 5, "gpu": 2}
    analysis = tune.run(
        tune_trainer, config=config, resources_per_trial=resources_per_trial
    )
    validate_analysis(analysis, token, config)
