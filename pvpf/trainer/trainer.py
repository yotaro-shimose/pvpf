from pathlib import Path
import time
from pvpf.property.training_property import TrainingProperty
from pvpf.model.convlstm import build_conv_lstm
from pvpf.tfrecord.high_level import load_dataset
import tensorflow as tf
from ray import tune
from typing import TypedDict
from ray.tune.integration.keras import TuneReportCheckpointCallback
import os


class TrainingConfig(TypedDict):
    batch_size: int
    num_epochs: int
    learning_rate: float
    training_property: TrainingProperty
    cwd: Path


def tune_trainer(config: TrainingConfig) -> None:
    os.chdir(config["cwd"])
    model = build_conv_lstm()
    train_x, test_x, train_y, test_y = load_dataset(config["training_property"])
    train_dataset = tf.data.Dataset.zip((train_x, train_y))
    test_dataset = tf.data.Dataset.zip((test_x, test_y))
    train_dataset = train_dataset.batch(config["batch_size"])
    train_dataset = train_dataset.shuffle(100 * 24, reshuffle_each_iteration=True)
    test_dataset = test_dataset.batch(config["batch_size"])

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    callbacks = list()
    tune_report_callback = TuneReportCheckpointCallback()
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=1, mode="min"
    )
    callbacks.append(tune_report_callback)
    callbacks.append(early_stopping_callback)
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=config["num_epochs"],
        callbacks=callbacks,
    )
