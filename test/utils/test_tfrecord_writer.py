import numpy as np
import tensorflow as tf
import os
from pvpf.utils.tfrecord_writer import write_tfrecord, read_tfrecord


def test_tfrecord_writer():
    features = np.random.normal(loc=np.zeros((3, 3, 3)))
    targets = np.random.normal(loc=np.zeros((3,)))
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    file_name = "test.tfrecord"
    write_tfrecord(file_name, dataset)
    loaded_dataset = read_tfrecord(file_name)
    loaded_features = list()
    loaded_targets = list()
    for feature, target in loaded_dataset:
        loaded_features.append(feature.numpy())
        loaded_targets.append(target.numpy())
    os.remove(file_name)
    loaded_features = np.stack(loaded_features)
    loaded_targets = np.stack(loaded_targets)
    assert np.all(np.isclose(features, loaded_features))
    assert np.all(np.isclose(targets, loaded_targets))
