import shutil
from datetime import datetime, timedelta
from pathlib import Path

import tensorflow as tf
from pvpf.preprocessor.cycle_encoder import CycleEncoder
from pvpf.property.dataset_property import DatasetProperty
from pvpf.property.tfrecord_property import TFRecordProperty
from pvpf.tfrecord.high_level import create_tfrecord, load_dataset
from pvpf.tfrecord.io import load_tfrecord, write_tfrecord

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


def test_load_dataset():
    image_shape = ()
    preprocessors = list()
    prop = TFRecordProperty(
        "test",
        "range",
        timedelta(hours=1),
        ["number"],
        image_shape,
        datetime(2020, 4, 1, 0, 0, 0),
        datetime(2020, 4, 2, 0, 0, 0),
        preprocessors,
    )
    path = Path(".").joinpath("tfrecords", "test-range")
    feature_path1 = path.joinpath("feature", "segment0.tfrecord")
    feature_path2 = path.joinpath("feature", "segment1.tfrecord")
    feature_dataset1 = tf.data.Dataset.range(12)
    feature_dataset2 = tf.data.Dataset.range(12, 24)
    feature_dataset1 = feature_dataset1.map(lambda x: tf.reshape(x, (1,)))
    feature_dataset2 = feature_dataset2.map(lambda x: tf.reshape(x, (1,)))
    write_tfrecord(feature_path1, feature_dataset1)
    write_tfrecord(feature_path2, feature_dataset2)
    target_path1 = path.joinpath("target", "segment0.tfrecord")
    target_path2 = path.joinpath("target", "segment1.tfrecord")
    target_dataset1 = tf.data.Dataset.range(12)
    target_dataset2 = tf.data.Dataset.range(12, 24)
    write_tfrecord(target_path1, target_dataset1)
    write_tfrecord(target_path2, target_dataset2)
    delta = 2
    window = 3
    ds_prop = DatasetProperty(
        prop,
        datetime(2020, 4, 1, 5, 0, 0),
        datetime(2020, 4, 1, 10, 0, 0),
        datetime(2020, 4, 1, 16),
        delta,
        window=window,
    )
    train_feature, test_feature, train_target, test_target = load_dataset(ds_prop)
    for i, feature, target in zip(range(5, 10), train_feature, train_target):
        assert feature.numpy().tolist() == [
            [i - (window - 1) - delta + j] for j in range(window)
        ]
        assert target.numpy().tolist() == i
    for i, feature, target in zip(range(10, 16), test_feature, test_target):
        assert feature.numpy().tolist() == [
            [i - (window - 1) - delta + j] for j in range(window)
        ]
        assert target.numpy().tolist() == i

    shutil.rmtree(path)


def test_load_dataset_with_mask():
    image_shape = ()
    preprocessors = list()
    prop = TFRecordProperty(
        "test",
        "range",
        timedelta(hours=1),
        ["number"],
        image_shape,
        datetime(2020, 4, 1, 0, 0, 0),
        datetime(2020, 4, 2, 0, 0, 0),
        preprocessors,
    )
    path = Path(".").joinpath("tfrecords", "test-range")
    if path.exists():
        shutil.rmtree(path)
    feature_path1 = path.joinpath("feature", "segment0.tfrecord")
    feature_path2 = path.joinpath("feature", "segment1.tfrecord")
    feature_dataset1 = tf.data.Dataset.range(12)
    feature_dataset2 = tf.data.Dataset.range(12, 24)
    feature_dataset1 = feature_dataset1.map(lambda x: tf.reshape(x, (1,)))
    feature_dataset2 = feature_dataset2.map(lambda x: tf.reshape(x, (1,)))
    write_tfrecord(feature_path1, feature_dataset1)
    write_tfrecord(feature_path2, feature_dataset2)
    target_path1 = path.joinpath("target", "segment0.tfrecord")
    target_path2 = path.joinpath("target", "segment1.tfrecord")
    target_dataset1 = tf.data.Dataset.range(12)
    target_dataset2 = tf.data.Dataset.range(12, 24)
    write_tfrecord(target_path1, target_dataset1)
    write_tfrecord(target_path2, target_dataset2)
    delta = 2
    window = 3
    datetime_mask = [
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        False,
        True,
        True,
        True,
    ]
    ds_prop = DatasetProperty(
        prop,
        datetime(2020, 4, 1, 5, 0, 0),
        datetime(2020, 4, 1, 10, 0, 0),
        datetime(2020, 4, 1, 16, 0, 0),
        delta,
        window=window,
        datetime_mask=datetime_mask,
    )
    train_feature, test_feature, train_target, test_target = load_dataset(ds_prop)
    masked_range = filter(lambda idx: datetime_mask[idx - 5], range(5, 10))
    for i, feature, target in zip(masked_range, train_feature, train_target):
        assert feature.numpy().tolist() == [
            [i - (window - 1) - delta + j] for j in range(window)
        ]
        assert target.numpy().tolist() == i
    masked_range = filter(lambda idx: datetime_mask[idx - 5], range(10, 16))
    for i, feature, target in zip(masked_range, test_feature, test_target):
        assert feature.numpy().tolist() == [
            [i - (window - 1) - delta + j] for j in range(window)
        ]
        assert target.numpy().tolist() == i

    shutil.rmtree(path)


def test_create_tfrecord():
    image_shape = (200, 200)
    preprocessors = list()
    preprocessors.append(CycleEncoder("datetime"))
    prop = TFRecordProperty(
        "test",
        "apbank",
        timedelta(hours=1),
        feature_names,
        image_shape,
        datetime(2020, 4, 1, 0, 0, 0),
        datetime(2020, 4, 3, 0, 0, 0),
        preprocessors,
    )
    path = create_tfrecord(prop)
    feature_path = path.joinpath("feature")
    target_path = path.joinpath("target")
    feature_dataset = load_tfrecord(feature_path)
    target_dataset = load_tfrecord(target_path)
    feature_count = target_count = 0
    for feature in feature_dataset:
        assert feature.shape == (200, 200, len(prop.feature_names))
        feature_count += 1
    for target in target_dataset:
        assert target.shape == ()
        target_count += 1
    assert feature_count == target_count == 48
    shutil.rmtree(path)
