import numpy as np
import tensorflow.keras as keras
from pathlib import Path
from pvpf.model.convlstm import build_conv_lstm, ConvLSTM
import shutil
import tensorflow as tf


def test_build_conv_lstm_ioshape():
    model = build_conv_lstm()
    B = 8
    T = 5
    W = H = 200
    C = 1
    inputs = tf.random.normal(shape=(B, T, W, H, C))
    outputs = model(inputs)
    assert outputs.shape == (B,)


def test_convlstm_ioshape():
    num_layers = 3
    num_filters = 64
    output_scale = 1000
    dim_cushion = 64
    model = ConvLSTM(
        num_layers=num_layers,
        num_filters=num_filters,
        output_scale=output_scale,
        dim_cushion=dim_cushion,
    )
    B = 128
    T = 5
    W = H = 50
    C = 1
    inputs = tf.random.normal(shape=(B, T, W, H, C))
    outputs = model(inputs)
    assert outputs.shape == (B,)


def test_convlstm_save_load():
    path = Path(".").joinpath("test_save_dir")
    num_layers = 3
    num_filters = 64
    output_scale = 1000
    dim_cushion = 64
    model = ConvLSTM(
        num_layers=num_layers,
        num_filters=num_filters,
        output_scale=output_scale,
        dim_cushion=dim_cushion,
    )
    B = 128
    T = 5
    W = H = 50
    C = 1
    inputs = tf.random.normal(shape=(B, T, W, H, C))
    outputs = model(inputs)
    model.save(path)
    loaded_model = keras.models.load_model(path)
    loaded_outputs = loaded_model(inputs)
    assert np.all(outputs.numpy() == loaded_outputs.numpy())
    shutil.rmtree(path)
