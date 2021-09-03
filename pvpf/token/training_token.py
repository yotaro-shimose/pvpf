from datetime import datetime
from pvpf.token.tfrecord_token import TFRECORD_TOKENS
from pvpf.property.training_property import TrainingProperty


base_tfrecord_token = TFRECORD_TOKENS["base"]
base_token = TrainingProperty(
    base_tfrecord_token,
    datetime(2020, 4, 1, 4, 0, 0),
    datetime(2021, 1, 1, 0, 0, 0),
    datetime(2021, 4, 1, 0, 0, 0),
    delta=1,
    window=3,
)


TRAINING_TOKENS = {"base": base_token}
