import numpy as np


def compute_error_rate(pred: np.ndarray, target: np.ndarray) -> float:
    error = np.abs(pred - target)
    return np.sum(error) / np.sum(target)


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    return rmse
