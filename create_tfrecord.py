from pvpf.tfrecord.high_level import create_tfrecord
from pvpf.token.tfrecord_token import TFRECORD_TOKENS


if __name__ == "__main__":
    token = TFRECORD_TOKENS["small"]
    create_tfrecord(token)
