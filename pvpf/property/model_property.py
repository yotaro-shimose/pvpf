from typing import List, TypedDict

from pvpf.model.convlstm import ConvLSTMBlockParam


class ModelProperty(TypedDict):
    pass


class ConvLSTMRegressorProperty(ModelProperty):
    block_params: List[ConvLSTMBlockParam]


class TwoImageRegressorProperty(ModelProperty):
    lfm_block_params: List[ConvLSTMBlockParam]
    jaxa_block_params: List[ConvLSTMBlockParam]
