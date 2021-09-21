from pvpf.trainer.trainer import tune_trainer, TrainingConfig
from ray import tune
from pvpf.token.training_token import TRAINING_TOKENS
from pathlib import Path

if __name__ == "__main__":
    token = TRAINING_TOKENS["base"]
    config = TrainingConfig(
        batch_size=8,
        num_epochs=30,
        learning_rate=tune.grid_search([1e-3, 5e-4]),
        training_property=token,
        shuffle_buffer=(365 * 24),  # carefully set this value to avoid OOM
        cwd=Path(".").resolve(),
        save_freq=10,
    )
    resources_per_trial = {"cpu": 20, "gpu": 4}
    tune.run(tune_trainer, config=config, resources_per_trial=resources_per_trial)
