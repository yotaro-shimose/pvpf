from dataclasses import dataclass
from typing import List, TypedDict, TypeVar

from ray import tune
from ray.tune.sample import Domain

from pvpf.property.dataset_property import DatasetProperty
from pvpf.token.dataset_token import DATASET_TOKENS
from pvpf.token.model_token import MODEL_TOKENS, ModelProperty
from pvpf.trainer.trainer import TrainingConfig, tune_trainer
from pvpf.validation.validate_analysis import validate_analysis

SearchSpace = TypeVar("SearchSpace", int, float, bool, Domain)


class ResourcesPerTrial(TypedDict):
    cpu: int
    gpu: int


@dataclass
class TrainingController:
    model_prop: ModelProperty
    feature_dataset_properties: List[DatasetProperty]
    target_dataset_property: DatasetProperty
    batch_size: SearchSpace
    num_epochs: SearchSpace
    learning_rate: SearchSpace
    shuffle_buffer: int
    resources_per_trial: ResourcesPerTrial

    def run(self):
        model_class, model_args = (
            self.model_prop.model_class,
            self.model_prop.model_args,
        )
        config = TrainingConfig(
            feature_dataset_properties=self.feature_dataset_properties,
            target_dataset_property=self.target_dataset_property,
            model_class=model_class,
            model_args=model_args,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            shuffle_buffer=self.shuffle_buffer,
        )
        resources_per_trial = self.resources_per_trial
        analysis = tune.run(
            tune_trainer, config=config, resources_per_trial=resources_per_trial
        )
        validate_analysis(
            analysis,
            config,
        )


def main():
    model_token = MODEL_TOKENS["two_image"]
    feature_dataset_properties = [
        DATASET_TOKENS["masked_small"],
        DATASET_TOKENS["masked_jaxa"],
    ]
    target_dataset_property = DATASET_TOKENS["masked_small"]
    controller = TrainingController(
        model_prop=model_token,
        feature_dataset_properties=feature_dataset_properties,
        target_dataset_property=target_dataset_property,
        batch_size=64,
        num_epochs=10,
        learning_rate=tune.grid_search([1e-3, 1e-2]),
        shuffle_buffer=500,  # carefully set this value to avoid OOM
        resources_per_trial={"cpu": 5, "gpu": 2},
    )
    controller.run()


if __name__ == "__main__":
    main()
