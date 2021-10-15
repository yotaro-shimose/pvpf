from pvpf.trainer.trainer import tune_trainer, TrainingConfig
from ray import tune
from pvpf.token.training_token import TRAINING_TOKENS, with_delta
from pathlib import Path

if __name__ == "__main__":
    token = TRAINING_TOKENS["base"]
    config = TrainingConfig(
        batch_size=6,
        num_epochs=30,
        learning_rate=5e-4,
        target_scale=1e-3,
        training_property=tune.grid_search(
            [with_delta(window) for window in [2, 3, 4, 5]]
        ),
        shuffle_buffer=500,  # carefully set this value to avoid OOM
        cwd=Path(".").resolve(),
        save_freq=10,
    )
    resources_per_trial = {"cpu": 20, "gpu": 4}
    tune.run(tune_trainer, config=config, resources_per_trial=resources_per_trial)
