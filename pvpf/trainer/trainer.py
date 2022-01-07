from pathlib import Path
from typing import Callable, List, TypedDict

import tensorflow.keras as keras
from pvpf.property.dataset_property import DatasetProperty
from pvpf.property.model_property import ModelArgs
from pvpf.utils.setup_dataset import setup_dataset
from ray.tune.integration.keras import TuneReportCheckpointCallback


class TrainingConfig(TypedDict):
    feature_dataset_properties: List[DatasetProperty]
    target_dataset_property: DatasetProperty
    model_builder: Callable[[ModelArgs], keras.Model]
    model_args: ModelArgs
    batch_size: int
    num_epochs: int
    learning_rate: float
    shuffle_buffer: int


def tune_trainer(config: TrainingConfig, checkpoint_dir: str = None):

    model = setup_model(config["model_builder"], config["model_args"], checkpoint_dir)
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
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=1
    )
    callbacks.append(early_stopping_callback)
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=config["num_epochs"],
        callbacks=callbacks,
    )


def setup_model(
    model_builder: Callable[[ModelArgs], keras.Model],
    model_args: ModelArgs,
    checkpoint_dir: str,
):
    if checkpoint_dir is not None:
        path = Path(checkpoint_dir).joinpath("checkpoint")
        model: keras.models.Model = keras.models.load_model(path)
    else:
        model = model_builder(model_args)
    return model
