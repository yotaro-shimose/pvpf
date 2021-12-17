from datetime import datetime
from typing import List

from pvpf.property.training_property import TrainingProperty
from pvpf.utils.date_range import date_range


def datetime_from_prop(prop: TrainingProperty) -> List[datetime]:
    ans = date_range(
        prop.prediction_start, prop.prediction_end, prop.tfrecord_property.time_unit
    )
    if prop.datetime_mask is not None:
        filtered = filter(lambda x: prop.datetime_mask[x[0]], enumerate(ans))
        ans = map(lambda x: x[1], filtered)
    ans = list(ans)
    return ans
