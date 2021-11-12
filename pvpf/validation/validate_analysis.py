from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from pvpf.constants import OUTPUT_ROOT
from pvpf.property.training_property import TrainingProperty
from pvpf.tfrecord.high_level import load_dataset
from pvpf.trainer.trainer import TrainingConfig
from pvpf.utils.indicator import compute_error_rate
from pvpf.validation.to_csv import to_csv
from ray.tune.analysis import ExperimentAnalysis


def get_prediction(
    model: keras.models.Model, dataset: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    ys = list()
    ts = list()
    count = 0
    for x, t in dataset:
        y = model(x)
        ys.append(y.numpy())
        ts.append(t.numpy())
        count += len(t.numpy())
    ys = np.concatenate(ys, 0)
    ts = np.concatenate(ts, 0)
    return ys, ts


def validate_analysis(
    analysis: ExperimentAnalysis, train_prop: TrainingProperty, config: TrainingConfig
):
    trials = analysis.trials
    checkpoints = [
        analysis.get_best_checkpoint(trial, metric="val_mae", mode="min")
        for trial in trials
    ]
    for checkpoint in checkpoints:
        validate_checkpoint(checkpoint, train_prop, config["batch_size"])


def validate_checkpoint(checkpoint: str, train_prop: TrainingProperty, batch_size: int):
    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.is_dir()

    model_path = checkpoint_path.joinpath("checkpoint")
    model = keras.models.load_model(model_path)

    train_x, test_x, train_y, test_y = load_dataset(train_prop)
    train_dataset = tf.data.Dataset.zip((train_x, train_y)).batch(batch_size=batch_size)
    test_dataset = tf.data.Dataset.zip((test_x, test_y)).batch(batch_size=batch_size)

    train_pred, train_target = get_prediction(model, train_dataset)
    test_pred, test_target = get_prediction(model, test_dataset)

    train_error = compute_error_rate(train_pred, train_target)
    test_error = compute_error_rate(test_pred, test_target)
    print(f"train_error: {train_error} test_error: {test_error}")

    file_name = checkpoint_path.parent.name + ".csv"
    output_path = OUTPUT_ROOT.joinpath(file_name)
    prediction = np.concatenate([train_pred, test_pred], axis=0)
    target = np.concatenate([train_target, test_target])

    to_csv(train_prop, output_path, prediction, target)
