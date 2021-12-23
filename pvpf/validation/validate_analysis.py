from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from pvpf.constants import OUTPUT_ROOT
from pvpf.property.dataset_property import DatasetProperty
from pvpf.trainer.trainer import TrainingConfig
from pvpf.utils.indicator import compute_error_rate
from pvpf.utils.setup_dataset import setup_dataset
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
    analysis: ExperimentAnalysis,
    config: TrainingConfig,
):
    feature_props = config["feature_dataset_properties"]
    target_prop = config["target_dataset_property"]
    trials = analysis.trials
    checkpoints = [
        analysis.get_best_checkpoint(trial, metric="val_mae", mode="min")
        for trial in trials
    ]
    for checkpoint in checkpoints:
        validate_checkpoint(
            checkpoint, feature_props, target_prop, config["batch_size"]
        )


def validate_checkpoint(
    checkpoint: str,
    feature_props: List[DatasetProperty],
    target_prop: DatasetProperty,
    batch_size: int,
):
    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.is_dir()

    model_path = checkpoint_path.joinpath("checkpoint")
    model = keras.models.load_model(model_path)

    train_dataset, test_dataset = setup_dataset(
        feature_props,
        target_prop,
        batch_size,
        shuffle_buffer_size=None,
    )

    train_pred, train_target = get_prediction(model, train_dataset)
    test_pred, test_target = get_prediction(model, test_dataset)

    train_error = compute_error_rate(train_pred, train_target)
    test_error = compute_error_rate(test_pred, test_target)
    print(f"train_error: {train_error} test_error: {test_error}")

    file_name = checkpoint_path.parent.name + ".csv"
    output_path = OUTPUT_ROOT.joinpath(file_name)
    prediction = np.concatenate([train_pred, test_pred], axis=0)
    target = np.concatenate([train_target, test_target])

    to_csv(target_prop, output_path, prediction, target)
