from typing import List
import pandas as pd
from pvpf.preprocessor.preprocessor import Preprocessor
import numpy as np


class CycleEncoder(Preprocessor):
    MONTH_COS = "month_cos"
    MONTH_SIN = "month_sin"
    HOUR_COS = "hour_cos"
    HOUR_SIN = "hour_sin"

    def __init__(self, input_label: str):
        self._input_label = input_label

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._input_label in df.columns
        df[self._input_label] = pd.to_datetime(df[self._input_label])
        months = pd.Series(df[self._input_label]).apply(lambda x: x.month)
        hours = pd.Series(df[self._input_label]).apply(lambda x: x.hour)
        df[self.MONTH_COS] = np.cos(2 * np.pi * months / 12)
        df[self.MONTH_SIN] = np.sin(2 * np.pi * months / 12)
        df[self.HOUR_COS] = np.cos(2 * np.pi * hours / 24)
        df[self.HOUR_SIN] = np.sin(2 * np.pi * hours / 24)
        df.pop(self._input_label)
        return df

    @property
    def input_label(self) -> str:
        return self._input_label

    @property
    def output_labels(self) -> List[str]:
        return [self.MONTH_COS, self.MONTH_SIN, self.HOUR_COS, self.HOUR_SIN]
