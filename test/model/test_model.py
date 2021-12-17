import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras as keras
from pvpf.model.convlstm import ConvLSTMBlockParam, ConvLSTMRegressor


@pytest.fixture
def temp_path() -> Path:
    path = Path(".").joinpath("test_save_dir")
    yield path
    if path.exists():
        shutil.rmtree(path)


def build_convlstm_regressor():
    block_params = [
        ConvLSTMBlockParam(num_filters=16, kernel_size=5, pooling=None),
        ConvLSTMBlockParam(num_filters=64, kernel_size=4, pooling=2),
        ConvLSTMBlockParam(num_filters=256, kernel_size=3, pooling=2),
        ConvLSTMBlockParam(num_filters=1024, kernel_size=2, pooling=2),
        ConvLSTMBlockParam(num_filters=1024, kernel_size=2, pooling=None),
    ]
    model = ConvLSTMRegressor(block_params)
    return model


@pytest.mark.parametrize(
    "model_builder",
    [
        build_convlstm_regressor,
    ],
)
def test_mode_ioshapel(model_builder: Callable[[], keras.Model]):
    model = model_builder()
    assert_model_ioshape(model)


def assert_model_ioshape(model: keras.Model):
    batch_size = 128
    time_window = 3
    image_size = 50
    dim_feature = 16
    input_shape = (batch_size, time_window, image_size, image_size, dim_feature)
    inputs = tf.random.normal(shape=input_shape)
    outputs = model(inputs)
    assert outputs.shape == (batch_size,)


@pytest.mark.parametrize(
    "model_builder",
    [
        build_convlstm_regressor,
    ],
)
def test_model_save_load(temp_path: Path, model_builder: Callable[[], keras.Model]):
    model = model_builder()
    batch_size = 128
    time_window = 3
    image_size = 50
    dim_feature = 16
    input_shape = (batch_size, time_window, image_size, image_size, dim_feature)
    inputs = tf.random.normal(shape=input_shape)
    outputs = model(inputs)
    model.save(temp_path)
    loaded_model = keras.models.load_model(temp_path)
    loaded_outputs = loaded_model(inputs)
    assert np.all(outputs.numpy() == loaded_outputs.numpy())
