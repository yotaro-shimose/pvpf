from dataclasses import dataclass
from typing import Optional
from pvpf.property.tfrecord_property import TFRecordProperty
from datetime import datetime, timedelta


@dataclass
class TrainingProperty:
    tfrecord_property: TFRecordProperty
    prediction_start: datetime
    prediction_split: datetime  # target dataset will be split into target[start: split], target[split: end]
    prediction_end: datetime
    delta: int
    window: Optional[int]

    def __post_init__(self):
        assert self.window != 1, "use None if you won't use time series data"
