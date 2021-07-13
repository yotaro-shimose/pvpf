import os
import shutil
from datetime import datetime, timedelta
from itertools import count
from pathlib import Path

import numpy as np
import tensorflow as tf
from pvpf.constants import ORIGINAL_IMAGE_SIZE
from pvpf.property.tfrecord_property import TFRecordProperty
from pvpf.utils.tfrecord_writer import (
    create_tfrecord,
    load_as_dataset,
    load_image_feature,
    load_image_features,
    load_targets,
    load_tfrecord,
    write_tfrecord,
)

feature_names = [
    "lo",
    "la",
    "tmp",
    "rh",
    "tcdc",
    "lcdc",
    "mcdc",
    "hcdc",
    "dswrf",
]


def test_tfrecord_writer():
    features = np.random.normal(loc=np.zeros((3, 3, 3)))
    targets = np.random.normal(loc=np.zeros((3,)))
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    dir = Path("./").joinpath("tfrecords", "test_tfrecord_writer")
    file_name = "test.tfrecord"
    path = str(dir.joinpath(file_name))
    write_tfrecord(path, dataset)
    loaded_dataset = load_tfrecord(str(dir))
    loaded_features = list()
    loaded_targets = list()
    for feature, target in loaded_dataset:
        loaded_features.append(feature.numpy())
        loaded_targets.append(target.numpy())
    shutil.rmtree(str(dir))
    loaded_features = np.stack(loaded_features)
    loaded_targets = np.stack(loaded_targets)
    assert np.all(np.isclose(features, loaded_features))
    assert np.all(np.isclose(targets, loaded_targets))


def test_load_image_feature():
    path = Path("./data/apbank/features/202008/2020-08-01T00:00:00+09:00.csv")
    data = load_image_feature(path, feature_names)
    assert data.shape == (301, 301, len(feature_names))


def test_load_image_features():
    plant_name = "apbank"
    start = datetime(2020, 9, 1, 0, 0, 0, 0)
    end = datetime(2020, 9, 2, 23, 0, 0, 0)

    features = load_image_features(plant_name, feature_names, start, end)
    delta = end - start
    hours = round(delta.total_seconds() / 3600)
    assert features.shape == (hours, *ORIGINAL_IMAGE_SIZE, len(feature_names))


def test_load_targets():
    plant_name = "apbank"
    start = datetime(2020, 9, 1, 0, 0, 0, 0)
    end = datetime(2020, 9, 2, 23, 0, 0, 0)
    targets = load_targets(plant_name, start, end)
    delta = end - start
    hours = round(delta.total_seconds() / 3600)
    assert targets.shape == (hours,)


def test_load_as_datasets():
    plant_name = "apbank"
    horizon = 1
    time_delta = timedelta(hours=horizon)
    start = datetime(2020, 9, 1, 0, 0, 0)
    end = datetime(2020, 9, 1, 23, 0, 0)
    image_size = (200, 200)
    dataset = load_as_dataset(
        plant_name, feature_names, start, end, time_delta, image_size
    )
    delta = end - start
    hours = round(delta.total_seconds() / 3600)
    assert len(dataset) == hours - horizon
    feature: np.ndarray
    target: np.ndarray
    feature, target = next(dataset.__iter__())
    assert feature.shape == image_size + (len(feature_names),)
    assert target.shape == ()


def test_create_tfrecord():
    plant_name = "apbank"
    horizon = 1
    time_delta = timedelta(hours=horizon)
    start = datetime(2020, 9, 1, 0, 0, 0)
    end = datetime(2020, 9, 9, 23, 0, 0)
    image_size = (200, 200)
    prop = TFRecordProperty(
        "test", plant_name, time_delta, feature_names, image_size, start, end
    )
    dir_name = create_tfrecord(prop)
    dataset = load_tfrecord(dir_name)
    delta = end - start
    hours = round(delta.total_seconds() / 3600)
    feature: np.ndarray
    target: np.ndarray
    counter = count()
    for feature, target in dataset:
        assert feature.shape == image_size + (len(feature_names),)
        assert target.shape == ()
        next(counter)
    assert next(counter) == hours - horizon
    shutil.rmtree(dir_name)
