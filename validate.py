from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from pvpf.tfrecord.high_level import load_dataset
from pvpf.token.dataset_token import DATASET_TOKENS
from pvpf.utils.date_range import date_range
from pvpf.utils.indicator import compute_error_rate


def get_prediction(
    model: keras.Model, dataset: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    ys = list()
    ts = list()
    count = 0
    for x, t in dataset:
        y = model(x)
        ys.append(y.numpy())
        ts.append(t.numpy())
        count += len(t.numpy())
    ys = np.concatenate(ys, 0)
    ts = np.concatenate(ts, 0)
    return ys, ts


def validate(trial_name: str):
    batch_size = 4
    output_path = Path(".").joinpath("output", trial_name)
    model_path = Path(".").joinpath("savedmodel", trial_name)
    model: keras.Model = keras.models.load_model(model_path)
    model.summary()
    ds_prop = DATASET_TOKENS["small"]
    train_x, test_x, train_y, test_y = load_dataset(ds_prop)
    train_dataset = tf.data.Dataset.zip((train_x, train_y)).batch(batch_size=batch_size)
    test_dataset = tf.data.Dataset.zip((test_x, test_y)).batch(batch_size=batch_size)
    train_pred, train_target = get_prediction(model, train_dataset)
    test_pred, test_target = get_prediction(model, test_dataset)
    train_error = compute_error_rate(train_pred, train_target)
    test_error = compute_error_rate(test_pred, test_target)
    prediction = np.concatenate([train_pred, test_pred], axis=0)
    target = np.concatenate([train_target, test_target])
    datetime = list(
        date_range(
            ds_prop.prediction_start,
            ds_prop.prediction_end,
            ds_prop.tfrecord_property.time_unit,
        )
    )
    df_dict = {"datetime": datetime, "prediction": prediction, "target": target}
    df = pd.DataFrame(df_dict)
    print(f"train_error: {train_error} test_error: {test_error}")
    df.to_csv(output_path)


if __name__ == "__main__":
    trial_name = "tune_trainer_4c974_00000"
    validate(trial_name)
    trial_name = "tune_trainer_4c974_00001"
    validate(trial_name)
    trial_name = "tune_trainer_4c974_00002"
    validate(trial_name)
    trial_name = "tune_trainer_4c974_00003"
    validate(trial_name)
