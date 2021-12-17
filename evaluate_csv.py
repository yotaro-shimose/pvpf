from pathlib import Path

import pandas as pd

from pvpf.token.dataset_token import TRAINING_TOKENS
from pvpf.utils.indicator import compute_error_rate, compute_rmse

prop = TRAINING_TOKENS["base"]
paths = list()
paths.append(Path(".").joinpath("output", "oct1_1.csv"))
for path in paths:
    df = pd.read_csv(path)
    df.loc[:, "datetime"] = pd.to_datetime(df.loc[:, "datetime"])
    train_pred = df.loc[df["datetime"] < prop.prediction_split, "prediction"]
    train_target = df.loc[df["datetime"] < prop.prediction_split, "target"]
    test_pred = df.loc[df["datetime"] >= prop.prediction_split, "prediction"]
    test_target = df.loc[df["datetime"] >= prop.prediction_split, "target"]
    train_error = compute_error_rate(train_pred, train_target)
    test_error = compute_error_rate(test_pred, test_target)
    rmse = compute_rmse(test_pred, test_target)
    print("*" * 10 + str(path) + "*" * 10)
    print(f"train_error: {train_error}")
    print(f"test_error: {test_error}")
    print(f"RMSE: {rmse}")
