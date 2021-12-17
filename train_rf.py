from pvpf.property.dataset_property import RFTrainingProperty
from pvpf.token.dataset_token import TRAINING_TOKENS
from pvpf.trainer.rf_trainer import train_rf

if __name__ == "__main__":
    token = TRAINING_TOKENS["rf_preaugumented"]
    image_size = (3, 3)
    prop = RFTrainingProperty(token, image_size)
    train_rf(prop)
