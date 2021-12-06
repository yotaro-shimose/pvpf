from typing import List
from pvpf.property.training_property import TrainingProperty
from pvpf.utils.date_range import date_range
from datetime import datetime


def datetime_from_prop(prop: TrainingProperty) -> List[datetime]:
    ans = date_range(
        prop.prediction_start, prop.prediction_end, prop.tfrecord_property.time_unit
    )
    if prop.datetime_mask is not None:
        ans = map(
            lambda _, dt: dt,
            filter(lambda idx, _: prop.datetime_mask[idx], enumerate(ans)),
        )
    ans = list(ans)
    return ans
