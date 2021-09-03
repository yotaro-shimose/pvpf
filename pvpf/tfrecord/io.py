import os
from datetime import datetime, timedelta
from pathlib import Path
from pvpf.utils.center_crop import center_crop
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from pvpf.preprocessor.preprocessor import Preprocessor
from pvpf.utils.date_range import date_range
import re


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _serialize_example(value):
    value = tf.cast(value, tf.float32)
    features = {
        "value": _bytes_feature(tf.io.serialize_tensor(value)),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def _tf_serialize_example(value):
    tf_string = tf.py_function(_serialize_example, [value], tf.string)
    return tf.reshape(tf_string, ())


def _parse_function(example_proto):
    feature_description = {
        "value": tf.io.FixedLenFeature(
            [],
            tf.string,
            default_value="",
        )
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    feature = tf.io.parse_tensor(example["value"], out_type=tf.float32)
    return feature


def _load_image_feature(
    path: Path, feature_names: Tuple[str, ...], preprocessors: List[Preprocessor]
) -> np.ndarray:
    df: pd.DataFrame = pd.read_csv(path)
    for preprocessor in preprocessors:
        df = preprocessor.process(df)
    features = df[feature_names]
    size = np.sqrt(features.shape[0]).astype(np.int)
    assert size * size == features.shape[0]
    raw_data = features.values.reshape(size, size, len(feature_names))
    return raw_data


def _load_image_features(
    plant_name: str,
    feature_names: Tuple[str, ...],
    start: datetime,
    end: datetime,
    preprocessors: List[Preprocessor],
) -> np.ndarray:
    ans = list()
    for datetime in date_range(start, end, timedelta(hours=1)):
        folder_name = f"{datetime.year}{datetime.month:02}"
        file_name = f"{datetime.year}-{datetime.month:02}-{datetime.day:02}T{datetime.hour:02}:00:00+09:00.csv"
        path = Path("./").joinpath(
            "data", plant_name, "features", folder_name, file_name
        )
        image = _load_image_feature(path, feature_names, preprocessors)
        ans.append(image)
    ans = np.stack(ans)
    return ans


def write_tfrecord(path: Path, dataset: tf.data.Dataset):
    serialized_dataset = dataset.map(_tf_serialize_example)
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True)
    writer = tf.data.experimental.TFRecordWriter(str(path))
    writer.write(serialized_dataset)


def load_tfrecord(dir_path: Path) -> tf.data.Dataset:
    # sort filenames using the number in the filename
    # basically filename should always be segment{D}.tfrecord where D is a number
    filenames = list(
        sorted(dir_path.iterdir(), key=lambda s: int(re.sub(r"\D", "", str(s)))),
    )
    raw_dataset = tf.data.TFRecordDataset(filenames)

    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


def _load_targets(plant_name: str, start: datetime, end: datetime) -> np.ndarray:
    path = Path("./").joinpath("data", plant_name, "targets", "target.csv")
    with path.open(mode="r") as f:
        df: pd.DataFrame = pd.read_csv(f)
    date = pd.to_datetime(df["datetime"])
    lower = start <= date
    upper = date < end
    ans = df[lower & upper]
    ans = ans["generated_energy"]
    return ans


def create_feature_dataset(
    plant_name: str,
    feature_names: Tuple[str, ...],
    start: datetime,
    end: datetime,
    image_size: Tuple[int, int],
    preprocessors: List[Preprocessor],
) -> tf.data.Dataset:
    features = _load_image_features(
        plant_name, feature_names, start, end, preprocessors
    )
    features = center_crop(features, image_size)
    dataset = tf.data.Dataset.from_tensor_slices(features)
    return dataset


def create_target_dataset(
    plant_name: str, start: datetime, end: datetime
) -> tf.data.Dataset:
    targets = _load_targets(plant_name, start, end)
    dataset = tf.data.Dataset.from_tensor_slices(targets)
    return dataset
