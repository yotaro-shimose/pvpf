from functools import reduce
from typing import List, Optional, Tuple

import tensorflow as tf
from pvpf.property.dataset_property import DatasetProperty
from pvpf.tfrecord.high_level import load_feature_dataset, load_target_dataset
from pvpf.utils.normalize_dataset import normalize_dataset


def setup_single_feature(
    feature_property: DatasetProperty,
    normalize: bool,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_x, test_x = load_feature_dataset(feature_property)
    if normalize:
        train_x, test_x = normalize_dataset(train_x, test_x)
    return train_x, test_x


def setup_feature_dataset(
    feature_properties: List[DatasetProperty],
    normalize: bool,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    feature_datasets = [
        setup_single_feature(prop, normalize) for prop in feature_properties
    ]
    if len(feature_datasets) > 1:
        trains, tests = list(zip(*feature_datasets))
        train_x, test_x = tf.data.Dataset.zip(trains), tf.data.Dataset.zip(tests)
    else:
        train_x, test_x = feature_datasets[0]
    return train_x, test_x


def setup_dataset(
    feature_properties: List[DatasetProperty],
    target_property: DatasetProperty,
    batch_size: int,
    shuffle_buffer_size: Optional[int],
    normalize: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    assert len(feature_properties) > 0
    train_x, test_x = setup_feature_dataset(feature_properties, normalize)
    train_y, test_y = load_target_dataset(target_property)
    train_dataset = tf.data.Dataset.zip((train_x, train_y))
    test_dataset = tf.data.Dataset.zip((test_x, test_y))
    if shuffle_buffer_size is not None:
        train_dataset = train_dataset.shuffle(
            shuffle_buffer_size, reshuffle_each_iteration=True
        )
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, test_dataset
