from datetime import datetime
from pvpf.token.tfrecord_token import TFRECORD_TOKENS
from pvpf.property.training_property import TrainingProperty
from datetime import timedelta

base_tfrecord_token = TFRECORD_TOKENS["base"]
base_token = TrainingProperty(
    base_tfrecord_token,
    datetime(2020, 4, 1, 4, 0, 0),
    datetime(2021, 1, 1, 0, 0, 0),
    datetime(2021, 4, 1, 0, 0, 0),
    delta=1,
    window=3,
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


TRAINING_TOKENS = {"base": base_token}
