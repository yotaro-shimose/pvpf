from typing import List, Type, TypedDict

import tensorflow.keras as keras
from pvpf.property.dataset_property import DatasetProperty
from pvpf.property.model_property import ModelProperty
from pvpf.utils.setup_dataset import setup_dataset
from ray.tune.integration.keras import TuneReportCheckpointCallback
import tensorflow as tf


class TrainingConfig(TypedDict):
    feature_dataset_properties: List[DatasetProperty]
    target_dataset_property: DatasetProperty
    model_class: Type[keras.Model]
    model_prop: ModelProperty
    batch_size: int
    num_epochs: int
    learning_rate: float
    shuffle_buffer: int


def tune_trainer(config: TrainingConfig, checkpoint_dir: str = None):
    model = setup_model(config["model_class"], config["model_prop"], checkpoint_dir)
    train_dataset, test_dataset = setup_dataset(
        config["feature_dataset_properties"],
        config["target_dataset_property"],
        config["batch_size"],
        config["shuffle_buffer"],
    )
    optimizer = keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    callbacks = list()
    tune_report_callback = TuneReportCheckpointCallback(frequency=1)
    callbacks.append(tune_report_callback)
    summarize(model, train_dataset)
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=config["num_epochs"],
        callbacks=callbacks,
    )


def setup_model(
    model_class: Type[keras.Model],
    model_prop: ModelProperty,
    checkpoint_dir: str,
):
    if checkpoint_dir is not None:
        model: keras.models.Model = keras.models.load_model(checkpoint_dir)
    else:
        model = model_class(**model_prop)
    return model


def summarize(model: keras.Model, train_dataset: tf.data.Dataset):
    model_input, _ = next(iter(train_dataset))
    model(model_input)
    model.summary()
