from pathlib import Path
import pandas as pd
from pvpf.utils.indicator import compute_error_rate, compute_rmse
from pvpf.token.training_token import TRAINING_TOKENS

prop = TRAINING_TOKENS["base"]
paths = list()
paths.append(Path(".").joinpath("output", "oct1_0.csv"))
paths.append(Path(".").joinpath("output", "oct1_1.csv"))
paths.append(Path(".").joinpath("output", "oct1_2.csv"))
for path in paths:
    df = pd.read_csv(path)
    df.loc[:, "datetime"] = pd.to_datetime(df.loc[:, "datetime"])
    test_pred = df.loc[df["datetime"] >= prop.prediction_split, "prediction"]
    test_target = df.loc[df["datetime"] >= prop.prediction_split, "target"]
    error_rate = compute_error_rate(test_pred, test_target)
    rmse = compute_rmse(test_pred, test_target)
    print("*" * 10 + str(path) + "*" * 10)
    print(f"Error Rate: {error_rate}")
    print(f"RMSE: {rmse}")
