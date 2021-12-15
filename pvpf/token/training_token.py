from datetime import datetime, timedelta
from typing import Dict

from pvpf.property.training_property import TrainingProperty
from pvpf.token.tfrecord_token import TFRECORD_TOKENS
from pvpf.utils.generate_mask import must_generate_daytime

base_tfrecord_token = TFRECORD_TOKENS["base"]
base_token = TrainingProperty(
    base_tfrecord_token,
    datetime(2020, 4, 1, 4, 0, 0),
    datetime(2021, 1, 1, 0, 0, 0),
    datetime(2021, 4, 1, 0, 0, 0),
    delta=1,
    window=3,
)
small_tfrecord_token = TFRECORD_TOKENS["small"]
small_token = TrainingProperty(
    small_tfrecord_token,
    datetime(2020, 4, 1, 4, 0, 0),
    datetime(2021, 4, 1, 0, 0, 0),
    datetime(2021, 7, 1, 0, 0, 0),
    delta=1,
    window=3,
)

start = datetime(2020, 4, 1, 4, 0, 0)
split = datetime(2021, 4, 1, 0, 0, 0)
end = datetime(2021, 7, 1, 0, 0, 0)
masked_small_token = TrainingProperty(
    small_tfrecord_token,
    start,
    split,
    end,
    delta=1,
    window=3,
    datetime_mask=must_generate_daytime(small_tfrecord_token, start, end),
)

test_token = TrainingProperty(
    small_tfrecord_token,
    datetime(2020, 4, 1, 4, 0, 0),
    datetime(2021, 4, 1, 10, 0, 0),
    datetime(2021, 4, 1, 15, 0, 0),
    delta=1,
    window=3,
)

rf_test_token = TrainingProperty(
    small_tfrecord_token,
    datetime(2021, 4, 1, 4, 0, 0),
    datetime(2021, 4, 1, 10, 0, 0),
    datetime(2021, 4, 1, 15, 0, 0),
    delta=1,
    window=1,
)

rf_token = TrainingProperty(
    small_tfrecord_token,
    datetime(2020, 4, 1, 4, 0, 0),
    datetime(2021, 4, 1, 10, 0, 0),
    datetime(2021, 7, 1, 0, 0, 0),
    delta=1,
    window=1,
)

rf_token_preaugumented = TrainingProperty(
    small_tfrecord_token,
    datetime(2020, 4, 1, 4, 0, 0),
    datetime(2021, 1, 1, 10, 0, 0),
    datetime(2021, 4, 1, 0, 0, 0),
    delta=1,
    window=1,
)


def with_delta(delta: int) -> TrainingProperty:
    window = 3
    token = TrainingProperty(
        base_tfrecord_token,
        datetime(2020, 4, 1, 0, 0, 0) + timedelta(hours=window + delta),
        datetime(2021, 1, 1, 0, 0, 0),
        datetime(2021, 4, 1, 0, 0, 0),
        delta=delta,
        window=3,
    )
    return token


TRAINING_TOKENS: Dict[str, TrainingProperty] = {
    "base": base_token,
    "small": small_token,
    "test": test_token,
    "rf_test": rf_test_token,
    "rf": rf_token,
    "rf_preaugumented": rf_token_preaugumented,
    "masked_small": masked_small_token,
}
