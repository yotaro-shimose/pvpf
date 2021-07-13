import numpy as np
from pvpf.model.model import ResNet


def test_model_ioshape():
    input_shape = (200, 200, 9)
    batch_size = 64
    model = ResNet(3, input_shape, 32, 3, 2, 512)
    input_shape = (batch_size,) + input_shape
    inputs = np.random.random(size=input_shape)
    outputs = model(inputs)
    assert outputs.shape == (batch_size,)
