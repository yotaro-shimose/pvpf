from typing import Dict
from pvpf.property.tfrecord_property import TFRecordProperty
from datetime import datetime, timedelta
from pvpf.preprocessor.cycle_encoder import CycleEncoder

base_token = TFRecordProperty(
    "base",
    "apbank",
    timedelta(hours=1),
    ("datetime", "tmp", "rh", "tcdc", "mcdc", "hcdc", "dswrf"),
    (200, 200),
    datetime(2020, 4, 1, 0, 0, 0),
    datetime(2021, 4, 1, 0, 0, 0),
    [CycleEncoder("datetime")],
)

TFRECORD_TOKENS: Dict[str, TFRecordProperty] = {"base": base_token}