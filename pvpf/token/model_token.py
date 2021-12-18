from dataclasses import dataclass
from typing import Dict, Type

import tensorflow.keras as keras
from pvpf.model.convlstm import ConvLSTMBlockParam, ConvLSTMRegressor, TwoImageRegressor
from pvpf.property.model_property import (
    ConvLSTMRegressorProperty,
    ModelProperty,
    TwoImageRegressorProperty,
)


@dataclass
class ModelToken:
    model_class: Type[keras.Model]
    model_prop: ModelProperty


conv_lstm = ModelToken(
    model_class=ConvLSTMRegressor,
    model_prop=ConvLSTMRegressorProperty(
        block_params=[
            ConvLSTMBlockParam(num_filters=16, kernel_size=5, pooling=None),
            ConvLSTMBlockParam(num_filters=64, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=128, kernel_size=3, pooling=2),
            ConvLSTMBlockParam(num_filters=256, kernel_size=2, pooling=None),
        ],
        l2=1e-3,
    ),
)

two_image = ModelToken(
    model_class=TwoImageRegressor,
    model_prop=TwoImageRegressorProperty(
        lfm_block_params=[
            ConvLSTMBlockParam(num_filters=16, kernel_size=5, pooling=None),
            ConvLSTMBlockParam(num_filters=64, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=128, kernel_size=3, pooling=2),
            ConvLSTMBlockParam(num_filters=256, kernel_size=2, pooling=None),
        ],
        jaxa_block_params=[
            ConvLSTMBlockParam(num_filters=16, kernel_size=5, pooling=None),
            ConvLSTMBlockParam(num_filters=64, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=128, kernel_size=3, pooling=2),
            ConvLSTMBlockParam(num_filters=256, kernel_size=2, pooling=None),
        ],
        l2=1e-3,
    ),
)

MODEL_TOKENS: Dict[str, ModelToken] = {
    "conv_lstm": conv_lstm,
    "two_image": two_image,
}
