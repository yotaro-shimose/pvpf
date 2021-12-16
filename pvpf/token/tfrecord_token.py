from datetime import datetime, timedelta
from typing import Dict

from pvpf.preprocessor.cycle_encoder import CycleEncoder
from pvpf.property.tfrecord_property import TFRecordProperty

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

small_token = TFRecordProperty(
    "small",
    "apbank",
    timedelta(hours=1),
    (
        "datetime",
        "prmsl",
        "pres",
        "ugrd",
        "vgrd",
        "tmp",
        "rh",
        "tcdc",
        "lcdc",
        "mcdc",
        "hcdc",
        "dswrf",
    ),
    (50, 50),
    datetime(2020, 4, 1, 0, 0, 0),
    datetime(2021, 7, 1, 0, 0, 0),
    [CycleEncoder("datetime")],
)

jaxa_token = TFRecordProperty(
    "jaxa",
    "apbank",
    timedelta(hours=1),
    (
        "CLOT",  # CLoud Optical Thickness
        "CLTT",  # Cloud Top Temperature
        "CLTH",  # Cloud Top Height
        # "CLER_23",
        # "CLTYPE",
        # "QA",
        "month_cos",
        "month_sin",
        "hour_cos",
        "hour_sin",
    ),
    (20, 20),
    datetime(2020, 4, 1, 0, 0, 0),
    datetime(2021, 7, 1, 0, 0, 0),
    [],
)


TFRECORD_TOKENS: Dict[str, TFRecordProperty] = {
    "base": base_token,
    "small": small_token,
    "jaxa": jaxa_token,
}
