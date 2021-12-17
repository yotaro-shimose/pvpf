from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from pvpf.property.tfrecord_property import TFRecordProperty
from pvpf.tfrecord.io import load_tfrecord
from pvpf.utils.dataset_to_numpy import dataset_to_numpy
from pvpf.utils.date_range import date_range


def _range_df(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    cond = (start <= df["datetime"]) & (df["datetime"] < end)
    return df.loc[cond, :]


def _entire_target(
    tfrecord_prop: TFRecordProperty, start: datetime, end: datetime
) -> pd.DataFrame:
    assert start >= tfrecord_prop.start and end <= tfrecord_prop.end, "invalid range"
    dir_path = tfrecord_prop.dir_path
    target_path = dir_path.joinpath("target")
    target = dataset_to_numpy(load_tfrecord(target_path))
    dt = list(
        date_range(tfrecord_prop.start, tfrecord_prop.end, tfrecord_prop.time_unit)
    )
    df = pd.DataFrame({"datetime": dt, "target": target})
    df["datetime"] = pd.to_datetime(df.loc[:, "datetime"])
    df = _range_df(df, start, end)
    return df


def must_generate_daytime(
    tfrecord_prop: TFRecordProperty, start: datetime, end: datetime
) -> List[bool]:
    ans = list()
    df = _entire_target(tfrecord_prop, start, end)
    start_day = datetime(start.year, start.month, start.day, 0, 0, 0)
    end_day = end + timedelta(days=1)
    end_day = datetime(end_day.year, end_day.month, end_day.day, 0, 0, 0)
    for am0 in date_range(start_day, end_day, timedelta(days=1)):
        pm0 = am0 + timedelta(days=1)
        day_df = _range_df(df, am0, pm0)
        am10 = datetime(am0.year, am0.month, am0.day, 10, 0, 0)
        pm4 = datetime(am0.year, am0.month, am0.day, 16, 0, 0)
        date_df = _range_df(day_df, am10, pm4)
        is_valid = not np.any(date_df.loc[:, "target"] == 0.0)
        ans.extend([is_valid for _ in range(len(day_df))])
    return ans
