import tensorflow as tf


def build_conv_lstm() -> tf.keras.Sequential:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ConvLSTM2D(
                filters=16, kernel_size=5, return_sequences=True
            ),
            # tf.keras.layers.LayerNormalization(),
            # tf.keras.layers.ConvLSTM2D(
            #     filters=64, kernel_size=5, padding="same", return_sequences=True
            # ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ConvLSTM2D(
                filters=64, kernel_size=5, padding="same", return_sequences=False
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Reshape(target_shape=()),
        ]
    )
    return model
