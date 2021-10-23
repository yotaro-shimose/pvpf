import tensorflow as tf
from pvpf.tfrecord.high_level import load_dataset
from pvpf.token.training_token import TRAINING_TOKENS
from pvpf.utils.indicator import compute_error_rate
from typing import Tuple
import numpy as np
from pvpf.utils.date_range import date_range
import pandas as pd
from pathlib import Path


def get_prediction(
    model: tf.keras.models.Model, dataset: tf.data.Dataset
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


if __name__ == "__main__":
    batch_size = 4
    output_paths = list()
    model_paths = list()
    output_paths.append(Path(".").joinpath("output", "oct1_1.csv"))
    model_paths.append(
        r"/root/workspace/pvpf/savedmodel/tune_trainer_eee50_00001"
    )  # 10/1_1
    output_paths.append(Path(".").joinpath("output", "oct1_2.csv"))
    model_paths.append(
        r"/root/workspace/pvpf/savedmodel/tune_trainer_eee50_00002"
    )  # 10/1_2
    for model_path, output_path in zip(model_paths, output_paths):
        model: tf.keras.models.Model = tf.keras.models.load_model(model_path)
        model.summary()
        train_prop = TRAINING_TOKENS["base"]
        train_x, test_x, train_y, test_y = load_dataset(train_prop)
        train_dataset = tf.data.Dataset.zip((train_x, train_y)).batch(
            batch_size=batch_size
        )
        test_dataset = tf.data.Dataset.zip((test_x, test_y)).batch(
            batch_size=batch_size
        )
        train_pred, train_target = get_prediction(model, train_dataset)
        test_pred, test_target = get_prediction(model, test_dataset)
        train_pred *= 1000
        test_pred *= 1000
        train_error = compute_error_rate(train_pred, train_target)
        test_error = compute_error_rate(test_pred, test_target)
        prediction = np.concatenate([train_pred, test_pred], axis=0)
        target = np.concatenate([train_target, test_target])
        datetime = list(
            date_range(
                train_prop.prediction_start,
                train_prop.prediction_end,
                train_prop.tfrecord_property.time_unit,
            )
        )
        df_dict = {"datetime": datetime, "prediction": prediction, "target": target}
        df = pd.DataFrame(df_dict)
        print(f"train_error: {train_error} test_error: {test_error}")
        df.to_csv(output_path)