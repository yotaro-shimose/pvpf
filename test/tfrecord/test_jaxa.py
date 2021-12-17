import shutil
from datetime import datetime, timedelta

from pvpf.property.tfrecord_property import TFRecordProperty
from pvpf.property.training_property import TrainingProperty
from pvpf.tfrecord.high_level import load_feature_dataset
from pvpf.tfrecord.jaxa import create_jaxa_tfrecord


def assert_create_jaxa_tfrecord(tfrecord_prop: TFRecordProperty):
    create_jaxa_tfrecord(tfrecord_prop)
    size = tfrecord_prop.image_size
    delta = 1
    start = datetime(2020, 4, 1, 5, 0, 0)
    num_train = 23
    split = start + timedelta(hours=num_train)
    num_test = 24
    end = split + timedelta(hours=num_test)
    window = 3
    train_prop = TrainingProperty(
        tfrecord_prop,
        start,
        split,
        end,
        delta,
        window,
    )
    train_dataset, test_dataset = load_feature_dataset(train_prop)
    train_count = 0
    for feature in train_dataset:
        train_count += 1
        assert feature.shape == (
            train_prop.window,
            *size,
            len(tfrecord_prop.feature_names),
        )
    assert train_count == num_train

    test_count = 0
    for feature in test_dataset:
        test_count += 1
        assert feature.shape == (
            train_prop.window,
            *size,
            len(tfrecord_prop.feature_names),
        )
    assert train_count == num_test


def test_create_jaxa_tfrecord():
    tfrecord_prop = TFRecordProperty(
        "jaxa_test",
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
        datetime(2020, 4, 3, 0, 0, 0),
        [],
    )
    try:
        assert_create_jaxa_tfrecord(tfrecord_prop)
    except Exception as e:
        print(e)
        if tfrecord_prop.dir_path.exists():
            shutil.rmtree(tfrecord_prop.dir_path)
