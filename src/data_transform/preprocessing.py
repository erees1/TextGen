import tensorflow as tf
from collections import Counter
import tensorflow_datasets as tfds
import numpy as np


def load_from_txt(f_path):
    dataset = tf.data.TextLineDataset(f_path)
    return dataset


def preprocess_data(
    dataset,
    buffer_size,
    batch_size,
    vocab_size=None,
    x_max_length=None,
    y_max_length=None,
    reverse_context=True,
    **kwargs,
):

    p_calls = tf.data.experimental.AUTOTUNE

    tokenizer = tfds.features.text.Tokenizer(reserved_tokens=['<sos>', '<eos>'], alphanum_only=True)

    # Build the vocab
    vocab_list = build_vocab_tfds(dataset, vocab_size, tokenizer, lowercase=True)
    integer_encoder = tfds.features.text.TokenTextEncoder(vocab_list, oov_token='<unk>', lowercase=True)

    def _filter_max_length(x, y):
        _x_max_length = x_max_length
        _y_max_length = y_max_length
        if x_max_length is None:
            _x_max_length = tf.size(x)
        if y_max_length is None:
            _y_max_length = tf.size(y)
        return tf.logical_and(tf.size(x) <= _x_max_length, tf.size(y) <= _y_max_length)

    def _encode(x, y):
        return encode(x, y, integer_encoder)

    # Seperate input and outputs
    dataset = dataset.map(split_labels)

    dataset = dataset.filter(_filter_max_length)

    dataset = dataset.map(
        lambda x, y: tf.py_function(_encode, [x, y], [tf.int32, tf.int32]),
        num_parallel_calls=p_calls,
    )

    dataset = dataset.map(lambda x, y: (x, y, y))

    dataset = dataset.map(split_outputs)

    if reverse_context:
        dataset = dataset.map(lambda x, x1, y: (x[::-1], x1, y))

    # dataset is fairly small (c. 200mb) so use the cache
    dataset = dataset.cache()

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size)

    # Padded batch
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None], [None]), drop_remainder=True)

    dataset = dataset.map(lambda x1, x2, y: ((x1, x2), y))

    # # Use prefetch to allow model to get batches in the background while training
    dataset.prefetch(p_calls)

    return dataset, integer_encoder


def build_vocab_tfds(dataset, max_size, tokenizer, lowercase=True):
    vocabulary_set = Counter()
    for text_tensor in dataset:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)
    if max_size is None:
        max_size = len(vocabulary_set)
    if lowercase:
        vocabulary_list = [i[0].lower() for i in vocabulary_set.most_common(max_size)]
    else:
        vocabulary_list = [i[0] for i in vocabulary_set.most_common(max_size)]

    return vocabulary_list


def build_vocab_keras(dataset, keras_tokenizer):
    text_list = [text.numpy().decode('utf-8') for text in dataset]
    print(text_list)
    print(type(text_list))
    keras_tokenizer.fit_on_texts(text_list)


# Map funtions
def split_labels(data):
    string = tf.strings.split(data, sep='--$--')
    if len(string) == 1:
        tf.print('error on', data)
    input_text = string[0]
    output_text = string[1]
    return input_text, output_text


def split_outputs(x, x2, y):
    return x, x2[:-1], y[1:]


def encode(input, output, encoder):
    # Only encode output - inputs will be encoded using pre trained embedding included in the model
    return encoder.encode(input.numpy()), encoder.encode(output.numpy())


def keras_encode(input1, output, encoder):
    return encoder.texts_to_sequences(input1), encoder.texts_to_sequences(input1)
