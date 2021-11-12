from dataclasses import dataclass
from typing import Optional, Tuple
from pvpf.property.tfrecord_property import TFRecordProperty
from datetime import datetime


@dataclass
class TrainingProperty:
    tfrecord_property: TFRecordProperty
    prediction_start: datetime
    prediction_split: datetime  # target dataset will be split into [start: split] and [split: end]
    prediction_end: datetime
    delta: int
    window: Optional[int]

    def __post_init__(self):
        assert (
            self.prediction_start >= self.tfrecord_property.start
            and self.prediction_end <= self.tfrecord_property.end
        ), "invalid prediction period"


@dataclass
class RFTrainingProperty:
    training_property: TrainingProperty
    image_size: Tuple[int, int]
