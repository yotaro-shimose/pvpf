from pvpf.trainer.rf_trainer import train_rf
from pvpf.token.training_token import TRAINING_TOKENS
from pvpf.property.training_property import RFTrainingProperty

if __name__ == "__main__":
    token = TRAINING_TOKENS["rf_preaugumented"]
    image_size = (3, 3)
    prop = RFTrainingProperty(token, image_size)
    train_rf(prop)
