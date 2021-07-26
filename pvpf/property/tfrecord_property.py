from dataclasses import MISSING, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from pvpf.preprocessor.preprocessor import Preprocessor


@dataclass
class TFRecordProperty:
    name: str
    plant_name: str
    time_delta: timedelta = timedelta(hours=1)
    feature_names: List[str] = field(
        default_factory=lambda: (
            "lo",
            "la",
            "tmp",
            "rh",
            "tcdc",
            "lcdc",
            "mcdc",
            "hcdc",
            "dswrf",
        ),
        default=MISSING,
    )
    image_size: Tuple[int, int] = (200, 200)
    start: datetime = datetime(2020, 8, 31, 0, 0, 0)
    end: datetime = datetime(2020, 11, 30, 23, 0, 0)
    preprocessors: List[Preprocessor] = field(default_factory=list, default=MISSING)

    @property
    def dir_name(self) -> str:
        dir_path = Path("./").joinpath("tfrecords", self.name, self.plant_name)
        return str(dir_path)

    def __post_init__(self):
        feature_names = set(self.feature_names)
        for processor in self.preprocessors:
            feature_names.remove(processor.input_label)
            for label in processor.output_labels:
                feature_names.add(label)
        self.feature_names = list(feature_names)
