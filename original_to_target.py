import pandas as pd
from pathlib import Path
import datetime
import numpy as np
from functools import reduce


def get_one_day_series(df: pd.DataFrame, date: datetime.date) -> pd.DataFrame:
    def get_hour_generation(hour) -> float:
        first_string = f"{hour:02}:00-{hour:02}:30"
        last_string = f"{hour:02}:30-{hour + 1:02}:00"
        record = df[df["datetime"].dt.date == date]
        return record[first_string] + record[last_string]

    datetimes = np.array(
        [datetime.datetime.combine(date, datetime.time(hour, 0)) for hour in range(24)]
    )
    generations = (
        np.array(list(map(get_hour_generation, range(24)))).squeeze(-1).astype(np.float)
    )
    ans = pd.DataFrame({"datetime": datetimes, "generated_energy": generations})
    return ans


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df.pop("date"))
    dates = df["datetime"].dt.date
    ans = reduce(
        lambda x, y: pd.merge(x, y, how="outer"),
        map(lambda x: get_one_day_series(df, x), dates),
    )
    return ans


input_path = Path("./").joinpath("temp.csv")

with input_path.open("r") as f:
    df: pd.DataFrame = pd.read_csv(f)

df_out = create_target(df)

output_path = Path("./").joinpath("temp_out.csv")

with output_path.open("w") as f:
    f.write(df_out.to_csv())
