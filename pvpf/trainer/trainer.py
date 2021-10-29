from pvpf.property.training_property import TrainingProperty
from pvpf.model.convlstm import ConvLSTM
from pvpf.tfrecord.high_level import load_dataset
import tensorflow as tf
import tensorflow.keras as keras
from typing import TypedDict
from ray.tune.integration.keras import TuneReportCheckpointCallback


class TrainingConfig(TypedDict):
    num_layers: int
    num_filters: int
    output_scale: float
    batch_size: int
    num_epochs: int
    learning_rate: float
    training_property: TrainingProperty
    shuffle_buffer: int


def tune_trainer(config: TrainingConfig, checkpoint_dir: str = None):
    if checkpoint_dir is not None:
        model: keras.models.Model = keras.models.load_model(checkpoint_dir)
    else:
        model = ConvLSTM(
            config["num_layers"], config["num_filters"], config["output_scale"]
        )
    train_x, test_x, train_y, test_y = load_dataset(config["training_property"])
    train_dataset = tf.data.Dataset.zip((train_x, train_y))
    test_dataset = tf.data.Dataset.zip((test_x, test_y))
    train_dataset = train_dataset.batch(config["batch_size"])
    train_dataset = train_dataset.shuffle(
        config["shuffle_buffer"], reshuffle_each_iteration=True
    )
    test_dataset = test_dataset.batch(config["batch_size"])

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
