from dataclasses import dataclass
from typing import List, TypedDict, TypeVar

from ray import tune
from ray.tune.sample import Domain
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

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
    use_bohb: bool = False
    num_samples: int = 1

    def run(self):
        model_builder, model_args = (
            self.model_prop.model_builder,
            self.model_prop.model_args,
        )
        config = TrainingConfig(
            feature_dataset_properties=self.feature_dataset_properties,
            target_dataset_property=self.target_dataset_property,
            model_builder=model_builder,
            model_args=model_args,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            shuffle_buffer=self.shuffle_buffer,
        )
        resources_per_trial = self.resources_per_trial

        metric = "val_mae"
        mode = "min"
        if self.use_bohb:
            time_attr = "training_iteration"
            algo = TuneBOHB(metric=metric, mode=mode)
            bohb = HyperBandForBOHB(
                time_attr=time_attr,
                metric=metric,
                mode=mode,
                max_t=self.num_epochs,
            )
            scheduler = bohb
            search_alg = algo
        else:
            scheduler = None
            search_alg = None

        analysis = tune.run(
            tune_trainer,
            config=config,
            resources_per_trial=resources_per_trial,
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=self.num_samples,
        )
        validate_analysis(
            analysis,
            config,
        )
        print(analysis.get_best_config(metric=metric, mode=mode))


def main():
    model_token = MODEL_TOKENS["conv_lstm_with_base_dim"]
    feature_dataset_properties = [
        DATASET_TOKENS["masked_small"],
        # DATASET_TOKENS["masked_jaxa"],
    ]
    target_dataset_property = DATASET_TOKENS["masked_small"]
    controller = TrainingController(
        model_prop=model_token,
        feature_dataset_properties=feature_dataset_properties,
        target_dataset_property=target_dataset_property,
        batch_size=64,
        num_epochs=50,
        learning_rate=tune.loguniform(1e-3, 3e-1),
        shuffle_buffer=500,  # carefully set this value to avoid OOM
        resources_per_trial={"cpu": 6, "gpu": 1},
        use_bohb=True,
        num_samples=64,
    )
    controller.run()


if __name__ == "__main__":
    main()
