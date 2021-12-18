from ray import tune

from pvpf.token.dataset_token import DATASET_TOKENS
from pvpf.token.model_token import MODEL_TOKENS
from pvpf.trainer.trainer import TrainingConfig, tune_trainer
from pvpf.validation.validate_analysis import validate_analysis

if __name__ == "__main__":
    model_token = MODEL_TOKENS["two_image"]
    model_class, model_prop = model_token.model_class, model_token.model_prop
    feature_dataset_properties = [
        DATASET_TOKENS["masked_small"],
        DATASET_TOKENS["masked_jaxa"],
    ]
    target_dataset_property = DATASET_TOKENS["masked_small"]
    config = TrainingConfig(
        feature_dataset_properties=feature_dataset_properties,
        target_dataset_property=target_dataset_property,
        model_class=model_class,
        model_prop=model_prop,
        batch_size=128,
        num_epochs=1,
        learning_rate=tune.grid_search([1e-4]),
        shuffle_buffer=500,  # carefully set this value to avoid OOM
    )
    resources_per_trial = {"cpu": 5, "gpu": 2}
    analysis = tune.run(
        tune_trainer, config=config, resources_per_trial=resources_per_trial
    )
    validate_analysis(
        analysis, feature_dataset_properties, target_dataset_property, config
    )
