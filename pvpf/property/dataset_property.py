from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from pvpf.property.tfrecord_property import TFRecordProperty
from pvpf.utils.date_range import date_range


@dataclass
class DatasetProperty:
    tfrecord_property: TFRecordProperty
    prediction_start: datetime
    prediction_split: datetime  # target dataset will be split into [start: split] and [split: end]
    prediction_end: datetime
    delta: int
    window: int
    datetime_mask: Optional[List[bool]] = None

    def __post_init__(self):
        assert (
            self.prediction_start - self.tfrecord_property.time_unit * self.delta
            >= self.tfrecord_property.start
            and self.prediction_end <= self.tfrecord_property.end
        ), "invalid prediction period"
        if self.datetime_mask is not None:
            assert len(
                list(
                    date_range(
                        self.prediction_start,
                        self.prediction_end,
                        self.tfrecord_property.time_unit,
                    )
                )
            ) == len(
                self.datetime_mask
            ), "datetime mask must have the same length as the number of target"


@dataclass
class RFTrainingProperty:
    dataset_property: DatasetProperty
    image_size: Tuple[int, int]
