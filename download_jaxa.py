from pvpf.tfrecord.jaxa import create_jaxa_tfrecord
from pvpf.token.tfrecord_token import jaxa_token

if __name__ == "__main__":
    create_jaxa_tfrecord(jaxa_token)
