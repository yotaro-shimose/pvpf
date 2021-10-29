import tensorflow.keras as keras
from pvpf.constants import KERAS_PACKAGE_NAME
import tensorflow as tf


@keras.utils.register_keras_serializable(package=KERAS_PACKAGE_NAME, name="ConvLSTM")
class ConvLSTM(keras.Model):
    def __init__(
        self, num_layers: int, num_filters: int, output_scale: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._num_layers = num_layers
        self._num_filters = num_filters
        self._output_scale = output_scale
        self._layers = list()
        for i in range(num_layers):
            self._layers.append(keras.layers.LayerNormalization())
            return_sequences = i + 1 != num_layers
            self._layers.append(
                keras.layers.ConvLSTM2D(
                    filters=num_filters,
                    kernel_size=5,
                    return_sequences=return_sequences,
                )
            )
        self._pooling = keras.layers.GlobalAveragePooling2D()
        self._final_dense = keras.layers.Dense(1)
        self._final_reshape = keras.layers.Reshape(target_shape=())

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for layer in self._layers:
            x = layer(x)
        x = self._pooling(x)
        x = self._final_dense(x)
        x = self._final_reshape(x)
        x = x * self._output_scale
        return x

    def get_config(self) -> dict:
        config: dict = super().get_config()
        custom_config = {
            "num_layers": self._num_layers,
            "num_filters": self._num_filters,
            "ouput_scale": self._output_scale,
        }
        config.update(custom_config)
        return config


def build_conv_lstm() -> keras.Sequential:
    model = keras.Sequential(
        [
            keras.layers.LayerNormalization(),
            keras.layers.ConvLSTM2D(filters=16, kernel_size=5, return_sequences=True),
            # keras.layers.LayerNormalization(),
            # keras.layers.ConvLSTM2D(
            #     filters=64, kernel_size=5, padding="same", return_sequences=True
            # ),
            keras.layers.LayerNormalization(),
            keras.layers.ConvLSTM2D(
                filters=64, kernel_size=5, padding="same", return_sequences=False
            ),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1),
            keras.layers.Reshape(target_shape=()),
        ]
    )
    return model
