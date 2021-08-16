from datetime import datetime, timedelta
from pathlib import Path
from pvpf.preprocessor.cycle_encoder import CycleEncoder
from typing import Tuple

import tensorflow as tf

from pvpf.model.model import ResNet
from pvpf.property.tfrecord_property import TFRecordProperty
from pvpf.utils.tfrecord_writer import create_tfrecord, load_tfrecord
from pvpf.utils.train_test_split import train_test_split


def get_base_prop(hours: int) -> TFRecordProperty:
    name = f"base-hours={hours}"
    plant_name = "apbank"
    time_delta = timedelta(hours=hours)
    feature_names = (
        "datetime",
        "tmp",
        "rh",
        "tcdc",
        "lcdc",
        "mcdc",
        "hcdc",
        "dswrf",
    )
    image_size = (200, 200)
    preprocessors = list()
    preprocessors.append(CycleEncoder("datetime"))
    start: datetime = datetime(2020, 4, 1, 0, 0, 0)
    end: datetime = datetime(2021, 4, 1, 0, 0, 0)
    prop = TFRecordProperty(
        name,
        plant_name,
        time_delta,
        feature_names,
        image_size,
        start,
        end,
        preprocessors,
    )
    return prop


def get_model(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    num_layers = 3
    base_channels = 32
    pool_size = 3
    strides = 2
    hidden_dim = 512
    model = ResNet(
        num_layers, input_shape, base_channels, pool_size, strides, hidden_dim
    )
    return model


def compute_error_rate(model: tf.keras.Model, dataset: tf.data.Dataset) -> tf.Tensor:
    predictions = list()
    targets = list()
    for x, y in dataset:
        prediction = model(x)
        predictions.append(prediction)
        targets.append(y)
    predictions = tf.concat(predictions, axis=-1)
    targets = tf.concat(targets, axis=-1)
    return tf.reduce_sum(tf.abs(predictions - targets)) / tf.reduce_sum(tf.abs(targets))


if __name__ == "__main__":
    log_dir = str(Path("./").joinpath("logs"))
    input_shape = (None, 200, 200, 11)
    num_epochs = 30
    batch_size = 128
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    errors = list()
    for hours in [3, 6, 9, 12]:
        prop = get_base_prop(hours)
        # create_tfrecord(prop)
        dataset = load_tfrecord(prop.dir_name)
        split = datetime(2021, 1, 1, 0, 0, 0)
        training_dataset, test_dataset = map(
            lambda x: x.batch(batch_size), train_test_split(dataset, prop, split)
        )
        model = get_model(input_shape)
        model.compile(optimizer="adam", loss="mae", metrics="mae")
        model.build(input_shape)

        model.fit(
            training_dataset,
            epochs=num_epochs,
            validation_data=test_dataset,
            callbacks=[tb_callback],
        )
        error_rate = compute_error_rate(model, test_dataset)
        errors.append(error_rate)
        print(error_rate)
    pass
