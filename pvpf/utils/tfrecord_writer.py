import tensorflow as tf


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _serialize_example(feature, target):
    feature = tf.cast(feature, tf.float32)
    target = tf.cast(target, tf.float32)
    features = {
        "feature": _bytes_feature(tf.io.serialize_tensor(feature)),
        "target": _bytes_feature(tf.io.serialize_tensor(target)),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def _tf_serialize_example(feature, target):
    tf_string = tf.py_function(_serialize_example, (feature, target), tf.string)
    return tf.reshape(tf_string, ())


def _parse_function(example_proto):
    feature_description = {
        "feature": tf.io.FixedLenFeature(
            [],
            tf.string,
            default_value="",
        ),
        "target": tf.io.FixedLenFeature(
            [],
            tf.string,
            default_value="",
        ),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    feature = tf.io.parse_tensor(example["feature"], out_type=tf.float32)
    target = tf.io.parse_tensor(example["target"], out_type=tf.float32)
    return feature, target


def write_tfrecord(file_name: str, dataset: tf.data.Dataset):
    serialized_dataset = dataset.map(_tf_serialize_example)
    writer = tf.data.experimental.TFRecordWriter(file_name)
    writer.write(serialized_dataset)


def read_tfrecord(file_name: str) -> tf.data.Dataset:
    filenames = [file_name]
    raw_dataset = tf.data.TFRecordDataset(filenames)

    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset
