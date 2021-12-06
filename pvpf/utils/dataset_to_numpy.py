import numpy as np
import tensorflow as tf


def dataset_to_numpy(dataset: tf.data.Dataset) -> np.ndarray:
    return np.stack([np.array(val) for val in dataset], axis=0)
