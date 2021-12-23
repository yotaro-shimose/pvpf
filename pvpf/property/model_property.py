from typing import List, TypedDict

from pvpf.model.convlstm import ConvLSTMBlockParam


class ModelArgs(TypedDict):
    pass


class ConvLSTMRegressorProperty(ModelArgs):
    block_params: List[ConvLSTMBlockParam]


class TwoImageRegressorProperty(ModelArgs):
    lfm_block_params: List[ConvLSTMBlockParam]
    jaxa_block_params: List[ConvLSTMBlockParam]
