import os
from datetime import datetime, timedelta
from pathlib import Path
from pvpf.preprocessor.preprocessor import Preprocessor
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from pvpf.property.tfrecord_property import TFRecordProperty
from pvpf.utils.converter import center_crop


def date_range(
    start: datetime, end: datetime, step: timedelta
) -> Generator[datetime, None, None]:
    cur = start
    while cur < end:
        yield cur
        cur += step


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _serialize_example(feature, target):
    feature = tf.cast(feature, tf.float32)
    target = tf.cast(target, tf.float32)
    features = {
        "feature": _bytes_feature(tf.io.serialize_tensor(feature)),
        "target": _bytes_feature(tf.io.serialize_tensor(target)),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def _tf_serialize_example(feature, target):
    tf_string = tf.py_function(_serialize_example, (feature, target), tf.string)
    return tf.reshape(tf_string, ())


def _parse_function(example_proto):
    feature_description = {
        "feature": tf.io.FixedLenFeature(
            [],
            tf.string,
            default_value="",
        ),
        "target": tf.io.FixedLenFeature(
            [],
            tf.string,
            default_value="",
        ),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    feature = tf.io.parse_tensor(example["feature"], out_type=tf.float32)
    target = tf.io.parse_tensor(example["target"], out_type=tf.float32)
    return feature, target


def write_tfrecord(path: Path, dataset: tf.data.Dataset):
    serialized_dataset = dataset.map(_tf_serialize_example)
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True)
    writer = tf.data.experimental.TFRecordWriter(str(path))
    writer.write(serialized_dataset)


def load_tfrecord(dir_name: str) -> tf.data.Dataset:
    filenames = list(
        map(lambda x: str(Path(dir_name).joinpath(x)), os.listdir(dir_name))
    )
    raw_dataset = tf.data.TFRecordDataset(filenames)

    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


def load_image_feature(
    path: Path, feature_names: Tuple[str, ...], preprocessors: List[Preprocessor]
) -> np.ndarray:
    with path.open(mode="r") as f:
        df: pd.DataFrame = pd.read_csv(f)
    for preprocessor in preprocessors:
        df = preprocessor.process(df)
    features = df[feature_names]
    size = np.sqrt(features.shape[0]).astype(np.int)
    raw_data = features.values.reshape(size, size, len(feature_names))
    return raw_data


def load_image_features(
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
        image = load_image_feature(path, feature_names, preprocessors)
        ans.append(image)
    ans = np.stack(ans)
    return ans


def load_targets(plant_name: str, start: datetime, end: datetime) -> np.ndarray:
    path = Path("./").joinpath("data", plant_name, "targets", "reshaped_data.csv")
    with path.open(mode="r") as f:
        df: pd.DataFrame = pd.read_csv(f)
    date = pd.to_datetime(df["datetime"])
    lower = start <= date
    upper = date < end
    ans = df[lower & upper]
    ans = ans["generated_energy"]
    return ans


def load_as_dataset(
    plant_name: str,
    feature_names: Tuple[str, ...],
    start: datetime,
    end: datetime,
    time_delta: timedelta,
    image_size: Tuple[int, int],
    preprocessors: List[Preprocessor],
) -> tf.data.Dataset:
    feature_start = start
    feature_end = end - time_delta
    target_start = start + time_delta
    target_end = end
    features = load_image_features(
        plant_name, feature_names, feature_start, feature_end, preprocessors
    )
    features = center_crop(features, image_size)
    targets = load_targets(plant_name, target_start, target_end)
    assert features.shape[0] == targets.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    return dataset


def create_tfrecord(prop: TFRecordProperty) -> str:
    window = timedelta(days=7)
    dir_path = Path("./").joinpath("tfrecords", prop.name, prop.plant_name)
    for index, start in enumerate(
        date_range(prop.start, prop.end, window - prop.time_delta)
    ):
        path = dir_path.joinpath(f"segment{index}.tfrecord")
        end = min(prop.end, start + window)
        dataset = load_as_dataset(
            prop.plant_name,
            prop.feature_names,
            start,
            end,
            prop.time_delta,
            prop.image_size,
            prop.preprocessors,
        )
        write_tfrecord(path, dataset)
    return dir_path
