from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple


@dataclass
class TFRecordProperty:
    name: str
    plant_name: str
    time_delta: timedelta = timedelta(hours=1)
    feature_names: Tuple[str, ...] = (
        "lo",
        "la",
        "tmp",
        "rh",
        "tcdc",
        "lcdc",
        "mcdc",
        "hcdc",
        "dswrf",
    )
    image_size: Tuple[int, int] = (200, 200)
    start: datetime = datetime(2020, 8, 31, 0, 0, 0)
    end: datetime = datetime(2020, 11, 30, 23, 0, 0)

    @property
    def dir_name(self) -> str:
        dir_path = Path("./").joinpath("tfrecords", self.name, self.plant_name)
        return str(dir_path)
