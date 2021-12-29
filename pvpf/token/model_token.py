from dataclasses import dataclass
from typing import Dict, Type

import tensorflow.keras as keras
from pvpf.model.convlstm import ConvLSTMBlockParam, ConvLSTMRegressor, TwoImageRegressor
from pvpf.property.model_property import (
    ConvLSTMRegressorProperty,
    ModelArgs,
    TwoImageRegressorProperty,
)
from ray import tune


@dataclass
class ModelProperty:
    model_class: Type[keras.Model]
    model_args: ModelArgs


conv_lstm = ModelProperty(
    model_class=ConvLSTMRegressor,
    model_args=ConvLSTMRegressorProperty(
        block_params=[
            ConvLSTMBlockParam(num_filters=16, kernel_size=5, pooling=None),
            ConvLSTMBlockParam(num_filters=64, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=128, kernel_size=3, pooling=2),
            ConvLSTMBlockParam(num_filters=256, kernel_size=2, pooling=None),
        ],
        l2=1e-3,
    ),
)

two_image = ModelProperty(
    model_class=TwoImageRegressor,
    model_args=TwoImageRegressorProperty(
        lfm_block_params=[
            ConvLSTMBlockParam(num_filters=16, kernel_size=5, pooling=None),
            ConvLSTMBlockParam(num_filters=32, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=32, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=64, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=64, kernel_size=3, pooling=4),
            ConvLSTMBlockParam(num_filters=64, kernel_size=2, pooling=None),
        ],
        jaxa_block_params=[
            ConvLSTMBlockParam(num_filters=16, kernel_size=5, pooling=None),
            ConvLSTMBlockParam(num_filters=32, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=32, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=64, kernel_size=3, pooling=4),
            ConvLSTMBlockParam(num_filters=64, kernel_size=2, pooling=None),
        ],
        l2=tune.loguniform(1e-1, 100),
    ),
)

MODEL_TOKENS: Dict[str, ModelProperty] = {
    "conv_lstm": conv_lstm,
    "two_image": two_image,
}
