from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
from pvpf.property.dataset_property import DatasetProperty
from pvpf.property.tfrecord_property import TFRecordProperty
from pvpf.tfrecord.io import (
    create_feature_dataset,
    create_target_dataset,
    load_tfrecord,
    write_tfrecord,
)
from pvpf.utils.count_hours import count_hours
from pvpf.utils.date_range import date_range


def _apply_mask(dataset: tf.data.Dataset, mask: List[bool]) -> tf.data.Dataset:
    mask = tf.constant(mask)
    return (
        dataset.enumerate()
        .filter(lambda idx, _: tf.gather(mask, idx))
        .map(lambda _, tensor: tensor)
    )


def _load_origin_target(ds_prop: DatasetProperty) -> tf.data.Dataset:
    target_shape = tuple()
    target_path = ds_prop.tfrecord_property.dir_path.joinpath("target")
    origin_target = load_tfrecord(target_path)
    origin_target = origin_target.map(lambda x: tf.ensure_shape(x, target_shape))
    return origin_target


def _load_origin_feature(ds_prop: DatasetProperty) -> tf.data.Dataset:
    feature_shape = ds_prop.tfrecord_property.image_size + (
        len(ds_prop.tfrecord_property.feature_names),
    )
    feature_path = ds_prop.tfrecord_property.dir_path.joinpath("feature")
    origin_feature = load_tfrecord(feature_path)
    origin_feature = origin_feature.map(lambda x: tf.ensure_shape(x, feature_shape))
    return origin_feature


def _split_mask(ds_prop: DatasetProperty) -> Tuple[List[bool], List[bool]]:
    num_train = count_hours(ds_prop.prediction_start, ds_prop.prediction_split)
    train_mask = ds_prop.datetime_mask[:num_train]
    test_mask = ds_prop.datetime_mask[num_train:]
    return train_mask, test_mask


def load_dataset(
    ds_prop: DatasetProperty,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_feature, test_feature = load_feature_dataset(ds_prop)
    train_target, test_target = load_target_dataset(ds_prop)
    return train_feature, test_feature, train_target, test_target


def load_feature_dataset(
    ds_prop: DatasetProperty,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    num_train = count_hours(ds_prop.prediction_start, ds_prop.prediction_split)
    num_test = count_hours(ds_prop.prediction_split, ds_prop.prediction_end)

    origin_feature = _load_origin_feature(ds_prop)
    batch_feature = origin_feature.window(size=ds_prop.window, shift=1).flat_map(
        lambda x: x.batch(ds_prop.window, drop_remainder=True)
    )
    num_init_skip = count_hours(
        ds_prop.tfrecord_property.start,
        ds_prop.prediction_start
        - ds_prop.delta * ds_prop.tfrecord_property.time_unit
        - (ds_prop.window - 1) * ds_prop.tfrecord_property.time_unit,
    )
    temp_batch_feature = batch_feature.skip(num_init_skip)
    train_feature = temp_batch_feature.take(num_train)
    test_feature = temp_batch_feature.skip(num_train).take(num_test)

    # apply mask
    if ds_prop.datetime_mask is not None:
        train_mask, test_mask = _split_mask(ds_prop)
        train_feature = _apply_mask(train_feature, train_mask)
        test_feature = _apply_mask(test_feature, test_mask)
    return train_feature, test_feature


def load_target_dataset(
    ds_prop: DatasetProperty,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    num_init_skip = count_hours(
        ds_prop.tfrecord_property.start, ds_prop.prediction_start
    )
    num_train = count_hours(ds_prop.prediction_start, ds_prop.prediction_split)
    num_test = count_hours(ds_prop.prediction_split, ds_prop.prediction_end)

    origin_target = _load_origin_target(ds_prop)
    temp_target = origin_target.skip(num_init_skip)
    train_target = temp_target.take(num_train)
    test_target = temp_target.skip(num_train).take(num_test)

    # apply mask
    if ds_prop.datetime_mask is not None:
        train_mask, test_mask = _split_mask(ds_prop)
        train_target = _apply_mask(train_target, train_mask)
        test_target = _apply_mask(test_target, test_mask)
    return train_target, test_target


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
