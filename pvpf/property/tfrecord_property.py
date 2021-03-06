from dataclasses import MISSING, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from pvpf.preprocessor.preprocessor import Preprocessor
from pvpf.constants import PROJECT_ROOT


@dataclass
class TFRecordProperty:
    name: str
    plant_name: str
    time_unit: timedelta = timedelta(hours=1)
    feature_names: List[str] = field(
        default_factory=lambda: (
            "datetime",
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
    start: datetime = datetime(2020, 4, 1, 0, 0, 0)
    end: datetime = datetime(2021, 4, 1, 0, 0, 0)
    preprocessors: List[Preprocessor] = field(default_factory=list, default=MISSING)

    @property
    def dir_path(self) -> Path:
        return PROJECT_ROOT.joinpath("tfrecords", f"{self.name}-{self.plant_name}")

    def __post_init__(self):
        feature_names = set(self.feature_names)
        for processor in self.preprocessors:
            feature_names.remove(processor.input_label)
            for label in processor.output_labels:
                feature_names.add(label)
        self.feature_names = list(sorted(feature_names))
