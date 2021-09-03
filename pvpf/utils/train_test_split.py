from datetime import datetime
from typing import Tuple

import tensorflow as tf
from pvpf.property.tfrecord_property import TFRecordProperty


def train_test_split(
    dataset: tf.data.Dataset, prop: TFRecordProperty, split_date: datetime
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    assert prop.start < split_date < prop.end
    # num_training := (end - start - delta) / delta + 1
    total_seconds = int((split_date - prop.start).total_seconds())
    assert total_seconds % prop.time_delta.total_seconds() == 0
    num_training = int(total_seconds // prop.time_delta.total_seconds())
    training_dataset = dataset.take(num_training)
    test_dataset = dataset.skip(num_training)
    return training_dataset, test_dataset
