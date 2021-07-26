from pvpf.preprocessor.cycle_encoder import CycleEncoder
import pandas as pd
import numpy as np
from pvpf.utils.tfrecord_writer import date_range
from datetime import datetime, timedelta


def test_cycle_encoder():
    EPS = 0.51
    cycle_name = "date"
    dummy_name = "dummy"
    cycle = pd.Series(
        np.array(
            list(
                date_range(
                    datetime(2021, 1, 1), datetime(2022, 1, 1), timedelta(hours=1)
                )
            ),
            dtype="datetime64[ns]",
        ),
    )
    dummy = pd.Series(np.random.random(size=len(cycle)), name=dummy_name)
    df = pd.DataFrame({cycle_name: cycle, dummy_name: dummy})
    preprocessor = CycleEncoder(cycle_name)
    df = preprocessor.process(df)
    labels = set(df.columns)
    assert "date" not in labels
    for label in preprocessor.output_labels:
        assert label in labels
        last_data = df[label].values[-1]
        for val in df[label].values:
            dif = abs(val - last_data)
            assert dif < EPS
            last_data = val
