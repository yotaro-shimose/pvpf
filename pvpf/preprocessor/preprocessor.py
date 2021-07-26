from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class Preprocessor(ABC):
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @property
    @abstractmethod
    def input_label(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_labels(self) -> List[str]:
        raise NotImplementedError
