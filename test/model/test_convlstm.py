from pvpf.model.convlstm import build_conv_lstm
import tensorflow as tf
import numpy as np


def test_build_conv_lstm_ioshape():
    model = build_conv_lstm()
    B = 8
    T = 5
    W = H = 200
    C = 1
    inputs1 = tf.random.normal(shape=(B, T, W, H, C))
    inputs2 = tf.random.normal(shape=(B, T, W, H, C))
    outputs1 = model(inputs1)
    outputs2 = model(inputs2)
    assert outputs1.shape == (B,)
    assert not np.all(np.isclose(outputs1.numpy(), outputs2.numpy()))
