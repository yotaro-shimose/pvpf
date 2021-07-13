from datetime import datetime
from typing import Tuple

import tensorflow as tf
from pvpf.property.tfrecord_property import TFRecordProperty


def train_test_split(
    dataset: tf.data.Dataset, prop: TFRecordProperty, split_date: datetime
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    assert prop.start < split_date < prop.end
    total_seconds = int((split_date - prop.start - prop.time_delta).total_seconds())
    assert total_seconds % 3600 == 0
    num_training = total_seconds // 3600
    training_dataset = dataset.take(num_training)
    test_dataset = dataset.skip(num_training)
    return training_dataset, test_dataset
