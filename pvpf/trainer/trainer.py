from typing import Callable, Tuple, TypedDict

import tensorflow as tf
import tensorflow.keras as keras
from pvpf.property.dataset_property import DatasetProperty
from pvpf.tfrecord.high_level import load_dataset
from pvpf.utils.normalize_dataset import normalize_dataset
from ray.tune.integration.keras import TuneReportCheckpointCallback


class ModelParam(TypedDict):
    pass


class TrainingConfig(TypedDict):
    model_builder: Callable[[ModelParam], keras.Model]
    model_param: ModelParam
    batch_size: int
    num_epochs: int
    learning_rate: float
    training_property: DatasetProperty
    shuffle_buffer: int


def tune_trainer(config: TrainingConfig, checkpoint_dir: str = None):
    model = setup_model(config["model_builder"], config["model_param"], checkpoint_dir)
    train_dataset, test_dataset = setup_dataset(
        config["training_property"], config["batch_size"], config["shuffle_buffer"]
    )
    optimizer = keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    callbacks = list()
    tune_report_callback = TuneReportCheckpointCallback(frequency=1)
    callbacks.append(tune_report_callback)
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=config["num_epochs"],
        callbacks=callbacks,
    )


def setup_model(
    model_builder: Callable[[ModelParam], keras.Model],
    param: ModelParam,
    checkpoint_dir: str,
):
    if checkpoint_dir is not None:
        model: keras.models.Model = keras.models.load_model(checkpoint_dir)
    else:
        model = model_builder(param)
    return model


def setup_dataset(
    prop: DatasetProperty,
    batch_size: int,
    shuffle_buffer_size: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_x, test_x, train_y, test_y = load_dataset(prop)
    train_x, test_x = normalize_dataset(train_x, test_x)
    train_dataset = tf.data.Dataset.zip((train_x, train_y))
    test_dataset = tf.data.Dataset.zip((test_x, test_y))
    train_dataset = train_dataset.shuffle(
        shuffle_buffer_size, reshuffle_each_iteration=True
    )
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, test_dataset
