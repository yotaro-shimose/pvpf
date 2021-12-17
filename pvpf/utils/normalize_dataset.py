from typing import Tuple

import tensorflow as tf


def _reduce_dataset(dataset: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor]:
    """calculate mean and std of dataset. only the last temporal data is used.

    Args:
        dataset (tf.data.Dataset): each element should have shape (T, H, W, F)

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: mean and std each have shape(1, H, W, F)
    """
    feature_shape = next(iter(dataset))[0].shape
    assert len(feature_shape) == 3, "dataset needs to have shape (T, H, W, F)"
    num_sample = 0
    mean = tf.zeros(feature_shape)
    mean_square = tf.zeros(feature_shape)
    for x in dataset:
        feature = x[-1]
        mean = mean / (num_sample + 1) * num_sample + feature / (num_sample + 1)
        mean_square = mean_square / (num_sample + 1) * num_sample + feature ** 2 / (
            num_sample + 1
        )
        num_sample += 1
    std = tf.sqrt(mean_square - mean ** 2)
    return tf.expand_dims(mean, 0), tf.expand_dims(std, 0)


def normalize_dataset(
    train_feature: tf.data.Dataset, test_feature: tf.data.Dataset
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    mean, std = _reduce_dataset(train_feature)
    train_feature = train_feature.map(lambda x: (x - mean) / std)
    test_feature = test_feature.map(lambda x: (x - mean) / std)
    return train_feature, test_feature
