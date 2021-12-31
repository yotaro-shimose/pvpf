from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import tensorflow.keras as keras
from pvpf.model.convlstm import ConvLSTMBlockParam, ConvLSTMRegressor, TwoImageRegressor
from pvpf.property.model_property import ModelArgs
from ray import tune


@dataclass
class ModelProperty:
    model_builder: Callable[[ModelArgs], keras.Model]
    model_args: ModelArgs


class ConvLSTMRegressorProperty(ModelArgs):
    block_params: List[ConvLSTMBlockParam]
    l2: Optional[float]
    dropout: float


class TwoImageRegressorProperty(ModelArgs):
    lfm_block_params: List[ConvLSTMBlockParam]
    jaxa_block_params: List[ConvLSTMBlockParam]
    l2: Optional[float]
    dropout: float


conv_lstm = ModelProperty(
    model_builder=lambda args: ConvLSTMRegressor(**args),
    model_args=ConvLSTMRegressorProperty(
        block_params=[
            ConvLSTMBlockParam(num_filters=16, kernel_size=5, pooling=None),
            ConvLSTMBlockParam(num_filters=32, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=64, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=128, kernel_size=4, pooling=2),
            ConvLSTMBlockParam(num_filters=256, kernel_size=3, pooling=4),
            ConvLSTMBlockParam(num_filters=1024, kernel_size=2, pooling=None),
        ],
        l2=1e-1,
        dropout=0.2,
    ),
)

two_image = ModelProperty(
    model_builder=lambda args: TwoImageRegressor(**args),
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
        dropout=tune.uniform(0.2, 0.6),
    ),
)


class BaseDim(ModelArgs):
    dim: int
    l2: float
    dropout: float


def conv_lstm_builder_with_base_dim(basedim: BaseDim) -> keras.Model:
    dim = basedim["dim"]
    l2 = basedim["l2"]
    dropout = basedim["dropout"]
    block_params = [
        ConvLSTMBlockParam(num_filters=dim * 1, kernel_size=5, pooling=None),
        ConvLSTMBlockParam(num_filters=dim * 2, kernel_size=4, pooling=2),
        ConvLSTMBlockParam(num_filters=dim * 4, kernel_size=4, pooling=2),
        ConvLSTMBlockParam(num_filters=dim * 8, kernel_size=4, pooling=2),
        ConvLSTMBlockParam(num_filters=dim * 16, kernel_size=3, pooling=4),
        ConvLSTMBlockParam(num_filters=dim * 64, kernel_size=2, pooling=None),
    ]
    l2 = l2
    dropout = dropout
    return ConvLSTMRegressor(block_params=block_params, l2=l2, dropout=dropout)


conv_lstm_with_base_dim = ModelProperty(
    model_builder=conv_lstm_builder_with_base_dim,
    model_args=BaseDim(
        dim=tune.qrandint(4, 16, q=2),
        l2=tune.loguniform(1e-2, 1e1),
        dropout=tune.uniform(0.0, 0.7),
    ),
)

MODEL_TOKENS: Dict[str, ModelProperty] = {
    "conv_lstm": conv_lstm,
    "two_image": two_image,
    "conv_lstm_with_base_dim": conv_lstm_with_base_dim,
}
