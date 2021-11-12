import pandas as pd
from pvpf.utils.date_range import date_range
from pvpf.property.training_property import TrainingProperty
from pathlib import Path
import numpy as np


def to_csv(
    train_prop: TrainingProperty, path: Path, prediction: np.ndarray, target: np.ndarray
):
    datetime = list(
        date_range(
            train_prop.prediction_start,
            train_prop.prediction_end,
            train_prop.tfrecord_property.time_unit,
        )
    )
    df_dict = {"datetime": datetime, "prediction": prediction, "target": target}
    df = pd.DataFrame(df_dict)

    df.to_csv(path)
