import numpy as np
import tensorflow as tf


def compute_error_rate(pred: np.ndarray, target: np.ndarray) -> float:
    error = np.abs(pred - target)
    return np.sum(error) / np.sum(target)


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    return rmse


def prediction_std(_target: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_std(pred)


def prediction_mean(_target: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_mean(pred)
