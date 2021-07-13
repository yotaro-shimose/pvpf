from datetime import datetime, timedelta
from typing import Tuple

import tensorflow as tf

from pvpf.model.model import ResNet
from pvpf.property.tfrecord_property import TFRecordProperty
from pvpf.utils.tfrecord_writer import create_tfrecord, load_tfrecord
from pvpf.utils.train_test_split import train_test_split


def get_base_prop() -> TFRecordProperty:
    name = "base"
    plant_name = "apbank"
    time_delta = timedelta(hours=1)
    feature_names = (
        "lo",
        "la",
        "tmp",
        "rh",
        "tcdc",
        "lcdc",
        "mcdc",
        "hcdc",
        "dswrf",
    )
    image_size = (200, 200)
    start: datetime = datetime(2020, 8, 31, 0, 0, 0)
    end: datetime = datetime(2020, 11, 1, 0, 0, 0)
    prop = TFRecordProperty(
        name, plant_name, time_delta, feature_names, image_size, start, end
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
    input_shape = (None, 200, 200, 9)
    num_epochs = 100
    batch_size = 128
    prop = get_base_prop()
    # create_tfrecord(prop)
    dataset = load_tfrecord(prop.dir_name)
    split = datetime(2020, 10, 1, 0, 0, 0)
    training_dataset, test_dataset = map(
        lambda x: x.batch(batch_size), train_test_split(dataset, prop, split)
    )
    model = get_model(input_shape)
    model.compile(optimizer="adam", loss="mae", metrics="mae")
    model.build(input_shape)
    model.fit(training_dataset, epochs=num_epochs, validation_data=test_dataset)
    error_rate = compute_error_rate(model, test_dataset)
    print(error_rate)
