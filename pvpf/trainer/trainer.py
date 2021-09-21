from pathlib import Path
from pvpf.property.training_property import TrainingProperty
from pvpf.model.convlstm import build_conv_lstm
from pvpf.tfrecord.high_level import load_dataset
import tensorflow as tf
from typing import TypedDict
from ray.tune.integration.keras import TuneReportCallback
import os
from ray import tune


class TrainingConfig(TypedDict):
    batch_size: int
    num_epochs: int
    learning_rate: float
    training_property: TrainingProperty
    shuffle_buffer: int
    cwd: Path
    save_freq: int


def tune_trainer(config: TrainingConfig, checkpoint_dir=None):
    os.chdir(config["cwd"])
    if checkpoint_dir is not None:
        model: tf.keras.models.Model = tf.keras.models.load_model(checkpoint_dir)
    else:
        model = build_conv_lstm()
    train_x, test_x, train_y, test_y = load_dataset(config["training_property"])
    train_dataset = tf.data.Dataset.zip((train_x, train_y))
    test_dataset = tf.data.Dataset.zip((test_x, test_y))
    train_dataset = train_dataset.batch(config["batch_size"])
    train_dataset = train_dataset.shuffle(
        config["shuffle_buffer"], reshuffle_each_iteration=True
    )
    test_dataset = test_dataset.batch(config["batch_size"])

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    callbacks = list()
    tune_report_callback = TuneReportCallback()
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=1, mode="min"
    )
    model_path = config["cwd"].joinpath("savedmodel", tune.get_trial_name())
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor="val_loss", save_best_only=True, mode="min"
    )
    callbacks.append(tune_report_callback)
    callbacks.append(early_stopping_callback)
    callbacks.append(save_model_callback)
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=config["num_epochs"],
        callbacks=callbacks,
    )
