import pandas as pd
from pathlib import Path
import numpy as np
from pvpf.utils.date_range import date_range
from datetime import datetime, timedelta

if __name__ == "__main__":
    path = Path(".").joinpath("newdata.csv")
    df = pd.read_csv(path)
    frame_labels = [f"frame{idx:02}" for idx in range(1, 48 + 1)]
    days_per_half_hour = df[frame_labels].values
    days_per_hour = [
        np.array([day[2 * i] + day[2 * i + 1] for i in range(24)])
        for day in days_per_half_hour
    ]
    hours = np.concatenate(days_per_hour, axis=0)
    datetimes = np.array(
        list(
            date_range(
                datetime(2021, 4, 1, 0, 0, 0),
                datetime(2021, 7, 1, 0, 0, 0),
                timedelta(hours=1),
            )
        )
    )
    assert len(hours) == len(datetimes)
    df_dict = {"datetime": datetimes, "generated_energy": hours}
    df = pd.DataFrame(df_dict)
    previous_path = Path(".").joinpath("data", "apbank", "targets", "target.csv")
    previous_df = pd.read_csv(previous_path, index_col=0)
    previous_df["datetime"] = pd.to_datetime(previous_df["datetime"])
    new_df = pd.concat([previous_df, df])
    out_path = Path(".").joinpath("data", "apbank", "targets", "new_target.csv")
    new_df.to_csv(out_path, index=False)
