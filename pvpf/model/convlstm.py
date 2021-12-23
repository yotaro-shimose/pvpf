from typing import List, Optional, Tuple, TypedDict

import tensorflow as tf
import tensorflow.keras as keras
from pvpf.constants import KERAS_PACKAGE_NAME


class ConvLSTMBlockParam(TypedDict):
    num_filters: int
    kernel_size: int
    pooling: int


@keras.utils.register_keras_serializable(
    package=KERAS_PACKAGE_NAME, name="ConvLSTMBlock"
)
class ConvLSTMBlock(keras.Model):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        pooling: int,
        return_sequences: Optional[bool],
        l2: Optional[float],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._pooling = pooling
        self._return_sequences = return_sequences
        self._layer_norm = keras.layers.LayerNormalization()
        if l2 is not None:
            regularizer = keras.regularizers.l2(l2)
        else:
            regularizer = None
        self._convlstm = keras.layers.ConvLSTM2D(
            num_filters,
            kernel_size,
            return_sequences=return_sequences,
            padding="same",
            kernel_regularizer=regularizer,
        )
        self._pooling = (
            keras.layers.MaxPool3D(pool_size=(1, pooling, pooling))
            if pooling is not None
            else None
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self._layer_norm(x)
        x = self._convlstm(x)
        if self._pooling is not None:
            x = self._pooling(x)
        return x

    def get_config(self) -> dict:
        default = super().get_config()
        ans = default | {
            "num_filters": self._num_filters,
            "kernel_size": self._kernel_size,
            "pooling": self._pooling,
            "return_sequences": self._return_sequences,
        }
        return ans


@keras.utils.register_keras_serializable(
    package=KERAS_PACKAGE_NAME, name="ConvLSTMImageEmbedder"
)
class ConvLSTMImageEmbedder(keras.Model):
    def __init__(
        self,
        block_params: List[ConvLSTMBlockParam],
        l2: Optional[float],
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self._block_params = block_params
        self._block_layers = [
            ConvLSTMBlock(
                **param, return_sequences=(idx != len(block_params) - 1), l2=l2
            )
            for idx, param in enumerate(block_params)
        ]
        self._flatten_layer = keras.layers.Flatten()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """spatio-temporal infomation into a single vector

        Args:
            x (tf.Tensor[B, T, W, H, F]): a spatio-temporal information

        Returns:
            tf.Tensor[B, F']: embedded infomation as a vector
        """
        for layer in self._block_layers:
            x = layer(x)
        x = self._flatten_layer(x)
        return x

    def get_config(self) -> dict:
        default = super().get_config()
        ans = default | {"block_params": self._block_params}
        return ans


@keras.utils.register_keras_serializable(
    package=KERAS_PACKAGE_NAME, name="ConvLSTMRegressor"
)
class ConvLSTMRegressor(keras.Model):
    def __init__(
        self,
        block_params: List[ConvLSTMBlockParam],
        l2: Optional[float],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._embedder = ConvLSTMImageEmbedder(block_params, l2)
        self._dense = keras.layers.Dense(1)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self._embedder(x)
        x = self._dense(x)
        x = tf.squeeze(x, -1)
        return x


@keras.utils.register_keras_serializable(
    package=KERAS_PACKAGE_NAME, name="TwoImageRegressor"
)
class TwoImageRegressor(keras.Model):
    def __init__(
        self,
        lfm_block_params: List[ConvLSTMBlockParam],
        jaxa_block_params: List[ConvLSTMBlockParam],
        l2: Optional[float],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._lfm_embedder = ConvLSTMImageEmbedder(lfm_block_params, l2)
        self._jaxa_embedder = ConvLSTMImageEmbedder(jaxa_block_params, l2)
        self._dense = keras.layers.Dense(1)

    def call(self, two_image: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        lfm, jaxa = two_image
        lfm = self._lfm_embedder(lfm)
        jaxa = self._jaxa_embedder(jaxa)
        embedding = tf.concat([lfm, jaxa], axis=-1)
        ans: tf.Tensor = self._dense(embedding)
        ans = tf.squeeze(ans, axis=-1)
        return ans
