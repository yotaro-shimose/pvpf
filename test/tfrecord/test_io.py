import shutil
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from pvpf.constants import ORIGINAL_IMAGE_SIZE
from pvpf.tfrecord.io import (
    _load_image_feature,
    _load_image_features,
    load_tfrecord,
    write_tfrecord,
)

feature_names = [
    "datetime",
    "tmp",
    "rh",
    "tcdc",
    "lcdc",
    "mcdc",
    "hcdc",
    "dswrf",
]


def test_load_image_feature():
    path = Path("./data/apbank/features/202004/2020-04-01T09:00:00+09:00.csv")
    data = _load_image_feature(path, feature_names, list())
    assert data.shape == (301, 301, len(feature_names))


def test_load_image_features():
    plant_name = "apbank"
    start = datetime(2020, 9, 1, 0, 0, 0, 0)
    end = datetime(2020, 9, 2, 23, 0, 0, 0)

    features = _load_image_features(plant_name, feature_names, start, end, list())
    delta = end - start
    hours = round(delta.total_seconds() / 3600)
    assert features.shape == (hours, *ORIGINAL_IMAGE_SIZE, len(feature_names))


def test_tfrecord_io():
    dir_path = Path(".").joinpath("tfrecords", "test")
    batch_size = 512
    num_iter = 3
    for i in range(num_iter):
        data = tf.random.normal((batch_size, 100, 100, 11))
        dataset = tf.data.Dataset.from_tensor_slices(data)
        file_path = dir_path.joinpath(f"test_{i}.tfrecord")
        write_tfrecord(file_path, dataset)
    loaded_dataset = load_tfrecord(dir_path)
    count = 0
    for val in loaded_dataset:
        count += 1
        assert val.shape == (100, 100, 11)
    assert count == num_iter * batch_size
    shutil.rmtree(dir_path)
