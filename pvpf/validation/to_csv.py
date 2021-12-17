import pandas as pd
from pvpf.property.training_property import TrainingProperty
from pathlib import Path
import numpy as np
from pvpf.utils.datetime_from_prop import datetime_from_prop


def to_csv(
    train_prop: TrainingProperty, path: Path, prediction: np.ndarray, target: np.ndarray
):
    datetime = datetime_from_prop(train_prop)
    df_dict = {"datetime": datetime, "prediction": prediction, "target": target}
    df = pd.DataFrame(df_dict)

    df.to_csv(path)
