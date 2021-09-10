from datetime import datetime, timedelta
from pathlib import Path
from pvpf.tfrecord.io import (
    create_feature_dataset,
    create_target_dataset,
    load_tfrecord,
    write_tfrecord,
)
from pvpf.property.tfrecord_property import TFRecordProperty

import tensorflow as tf
from pvpf.property.training_property import TrainingProperty
from pvpf.utils.date_range import date_range
from typing import Tuple


def _count_hours(start: datetime, end: datetime) -> int:
    dif = end - start
    assert dif.total_seconds() % 3600 == 0
    return int(dif.total_seconds() // 3600)


def load_dataset(
    train_prop: TrainingProperty,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    # target
    target_shape = tuple()
    target_path = train_prop.tfrecord_property.dir_path.joinpath("target")
    origin_target = load_tfrecord(target_path)
    origin_target = origin_target.map(lambda x: tf.ensure_shape(x, target_shape))

    num_init_skip = _count_hours(
        train_prop.tfrecord_property.start, train_prop.prediction_start
    )
    num_train = _count_hours(train_prop.prediction_start, train_prop.prediction_split)
    num_test = _count_hours(train_prop.prediction_split, train_prop.prediction_end)
    temp_target = origin_target.skip(num_init_skip)
    train_target = temp_target.take(num_train)
    test_target = temp_target.skip(num_train).take(num_test)

    # feature
    feature_shape = train_prop.tfrecord_property.image_size + (
        len(train_prop.tfrecord_property.feature_names),
    )
    feature_path = train_prop.tfrecord_property.dir_path.joinpath("feature")
    origin_feature = load_tfrecord(feature_path)
    origin_feature = origin_feature.map(lambda x: tf.ensure_shape(x, feature_shape))

    if train_prop.window is not None:
        batch_feature = origin_feature.window(size=train_prop.window, shift=1).flat_map(
            lambda x: x.batch(train_prop.window, drop_remainder=True)
        )
        num_init_skip = _count_hours(
            train_prop.tfrecord_property.start,
            train_prop.prediction_start
            - train_prop.delta * train_prop.tfrecord_property.time_unit
            - (train_prop.window - 1) * train_prop.tfrecord_property.time_unit,
        )
        temp_batch_feature = batch_feature.skip(num_init_skip)
        train_feature = temp_batch_feature.take(num_train)
        test_feature = temp_batch_feature.skip(num_train).take(num_test)
    else:
        raise NotImplementedError

    return train_feature, test_feature, train_target, test_target


def create_tfrecord(prop: TFRecordProperty) -> Path:
    window = timedelta(days=7)
    dir_path = prop.dir_path
    feature_path = dir_path.joinpath("feature")
    for index, start in enumerate(date_range(prop.start, prop.end, window)):
        file_path = feature_path.joinpath(f"segment{index}.tfrecord")
        end = min(prop.end, start + window)
        dataset = create_feature_dataset(
            prop.plant_name,
            prop.feature_names,
            start,
            end,
            prop.image_size,
            prop.preprocessors,
        )
        write_tfrecord(file_path, dataset)
    target_path = dir_path.joinpath("target", "segment0.tfrecord")
    dataset = create_target_dataset(prop.plant_name, prop.start, prop.end)
    write_tfrecord(target_path, dataset)
    return dir_path
