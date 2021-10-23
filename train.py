from pvpf.trainer.trainer import tune_trainer, TrainingConfig
from ray import tune
from pvpf.token.training_token import TRAINING_TOKENS
from pathlib import Path

if __name__ == "__main__":
    token = TRAINING_TOKENS["small"]
    config = TrainingConfig(
        batch_size=128,
        num_epochs=30,
        learning_rate=tune.grid_search([1e-4, 5e-4, 1e-3, 3e-3]),
        target_scale=1e-3,
        training_property=token,
        shuffle_buffer=500,  # carefully set this value to avoid OOM
        cwd=Path(".").resolve(),
        save_freq=10,
    )
    resources_per_trial = {"cpu": 5, "gpu": 1}
    tune.run(tune_trainer, config=config, resources_per_trial=resources_per_trial)
