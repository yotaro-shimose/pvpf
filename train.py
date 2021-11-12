from pvpf.trainer.trainer import tune_trainer, TrainingConfig
from ray import tune
from pvpf.token.training_token import TRAINING_TOKENS
from pvpf.validation.validate_analysis import validate_analysis

if __name__ == "__main__":
    token = TRAINING_TOKENS["small"]
    config = TrainingConfig(
        num_layers=3,
        num_filters=64,
        output_scale=1000,
        dim_cushion=256,
        batch_size=128,
        num_epochs=30,
        learning_rate=tune.grid_search([1e-4, 3e-4, 1e-3, 3e-3]),
        training_property=token,
        shuffle_buffer=500,  # carefully set this value to avoid OOM
    )
    resources_per_trial = {"cpu": 5, "gpu": 1}
    analysis = tune.run(
        tune_trainer, config=config, resources_per_trial=resources_per_trial
    )
    validate_analysis(analysis, token, config)
