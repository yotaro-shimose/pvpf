from typing import Callable

import tensorflow as tf


# Reference https://qiita.com/hima_zin331/items/2adba781bc4afaae5880
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        bneck_channels = out_channels // 4
        self._bn1 = tf.keras.layers.BatchNormalization()
        self._conv1 = tf.keras.layers.Conv2D(
            bneck_channels,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=False,
        )

        self._bn2 = tf.keras.layers.BatchNormalization()
        self._conv2 = tf.keras.layers.Conv2D(
            bneck_channels,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
        )

        self._bn3 = tf.keras.layers.BatchNormalization()
        self._conv3 = tf.keras.layers.Conv2D(
            out_channels,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=False,
        )

        self._shortcut = self._scblock(in_channels, out_channels)

    # Shortcut Connection
    def _scblock(
        self, in_channels: int, out_channels: int
    ) -> Callable[[tf.Tensor], tf.Tensor]:
        if in_channels != out_channels:
            self.bn_sc1 = tf.keras.layers.BatchNormalization()
            self.conv_sc1 = tf.keras.layers.Conv2D(
                out_channels, kernel_size=1, strides=1, padding="same", use_bias=False
            )
            return self.conv_sc1
        else:
            return lambda x: x

    def call(self, x: tf.Tensor) -> tf.Tensor:
        shortcut = self._shortcut(x)
        x = self._bn1(x)
        x = tf.keras.activations.relu(x)
        x = self._conv1(x)
        x = self._bn2(x)
        x = tf.keras.activations.relu(x)
        x = self._conv2(x)
        x = self._bn3(x)
        x = tf.keras.activations.relu(x)
        x = self._conv3(x)
        x = x + shortcut
        return x


# ResNet50(Pre Activation)
class ResNet(tf.keras.Model):
    def __init__(
        self,
        num_blocks: int,
        input_shape: int,
        base_channels: int,
        pool_size: int,
        strides: int,
        dim_hidden: int,
    ):
        super().__init__()
        self._layers = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(
                base_channels,
                kernel_size=7,
                strides=2,
                padding="same",
                use_bias=False,
                input_shape=input_shape,
            ),
            tf.keras.layers.MaxPool2D(
                pool_size=pool_size, strides=strides, padding="same"
            ),
            *[ResidualBlock(base_channels, base_channels) for _ in range(num_blocks)],
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(dim_hidden),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Dense(1),
        ]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for layer in self._layers:
            x = layer(x)
        x = tf.squeeze(x, axis=-1)
        return x
