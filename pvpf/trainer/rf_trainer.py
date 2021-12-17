from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from pvpf.constants import OUTPUT_ROOT
from pvpf.property.training_property import RFTrainingProperty
from pvpf.tfrecord.high_level import load_dataset
from pvpf.utils.center_crop import center_crop
from pvpf.utils.dataset_to_numpy import dataset_to_numpy
from pvpf.utils.indicator import compute_error_rate
from pvpf.validation.to_csv import to_csv
from sklearn.ensemble import RandomForestRegressor


def _create_trial_name() -> str:
    now = datetime.now()
    now_string = now.strftime("%Y:%m:%d-%H:%M:%S")
    ans = f"rf_{now_string}"
    return ans


def _center_crop_tf(tensor: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    cropped = center_crop(tensor, size)
    target_shape = (-1,)
    flattened = tf.reshape(tf.squeeze(cropped, axis=0), target_shape)
    return flattened


def _2d_feature_labels(feature_names: List[str], size: Tuple[int, int]) -> List[str]:
    ans = list()
    for i in range(size[0]):
        for j in range(size[1]):
            temp_features = [
                feature_name + f"_{i}-{j}" for feature_name in feature_names
            ]
            ans.extend(temp_features)
    return ans


def _importance_csv(
    prop: RFTrainingProperty, model: RandomForestRegressor, trial_dir: Path
):
    importance_file_name = "importance.csv"
    importance = model.feature_importances_
    feature_names = _2d_feature_labels(
        prop.training_property.tfrecord_property.feature_names, prop.image_size
    )
    importance_dict = {"feature_name": feature_names, "importance": importance}
    importance_path = trial_dir.joinpath(importance_file_name)
    importance_df = pd.DataFrame(importance_dict)
    importance_df.to_csv(importance_path)


def load_rf_dataset(
    prop: RFTrainingProperty,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_feature, test_feature, train_target, test_target = load_dataset(
        prop.training_property
    )
    train_feature = train_feature.map(
        lambda tensor: _center_crop_tf(tensor, prop.image_size)
    )
    test_feature = test_feature.map(
        lambda tensor: _center_crop_tf(tensor, prop.image_size)
    )
    datasets: List[tf.data.Dataset] = [
        train_feature,
        test_feature,
        train_target,
        test_target,
    ]

    datasets = tuple(dataset_to_numpy(dataset) for dataset in datasets)
    return datasets


def _get_trial_dir() -> Path:
    trial_name = _create_trial_name()
    trial_dir = OUTPUT_ROOT.joinpath(trial_name)
    if not trial_dir.exists():
        trial_dir.mkdir()
    return trial_dir


def validate_rf_result(
    prop: RFTrainingProperty,
    model: RandomForestRegressor,
    train_feature: np.ndarray,
    test_feature: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
):
    train_prediction = model.predict(train_feature)
    test_prediction = model.predict(test_feature)
    train_error_rate = compute_error_rate(train_prediction, train_target)
    test_error_rate = compute_error_rate(test_prediction, test_target)
    print(f"train_error: {train_error_rate}")
    print(f"test_error: {test_error_rate}")

    trial_dir = _get_trial_dir()

    prediction = np.concatenate([train_prediction, test_prediction], axis=0)
    target = np.concatenate([train_target, test_target], axis=0)
    output_file_name = "output.csv"
    output_path = trial_dir.joinpath(output_file_name)
    to_csv(prop.training_property, output_path, prediction, target)
    _importance_csv(prop, model, trial_dir)


def train_rf(prop: RFTrainingProperty):
    train_feature, test_feature, train_target, test_target = load_rf_dataset(prop)

    model = RandomForestRegressor()
    model.fit(train_feature, train_target)
    validate_rf_result(
        prop, model, train_feature, test_feature, train_target, test_target
    )
