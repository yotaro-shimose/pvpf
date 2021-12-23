from pathlib import Path

import numpy as np
import pandas as pd
from pvpf.property.dataset_property import DatasetProperty
from pvpf.utils.datetime_from_prop import datetime_from_prop


def to_csv(
    ds_prop: DatasetProperty, path: Path, prediction: np.ndarray, target: np.ndarray
):
    datetime = datetime_from_prop(ds_prop)
    df_dict = {"datetime": datetime, "prediction": prediction, "target": target}
    df = pd.DataFrame(df_dict)

    df.to_csv(path)
