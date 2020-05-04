# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from src.data.msg_pipeline import WhiteSpaceTokenizer, Padder, Word2Vec, IntegerTokenizer, Tagger
from sklearn.pipeline import Pipeline
from gensim.models import KeyedVectors
import logging


def process_data(file_path, batch_size, buffer_size, flip_context=True):
    """Function to perform pre-processing: shuffle, batch etc prior to modelling and return tf.data.dataset

    Arguments:
        file_path {string} -- path to the clean data
        batch_size {int} -- batch size for tf batch operation
        buffer_size {int} -- buffer size for tf shuffle operation

    Keyword Arguments:
        flip_context {bool} -- whether to reverse the context input (default: {True})

    Returns:
        processed dataset {tf.data.dataset}
    """
    fb_tokens = np.load(file_path)
    X = fb_tokens['X']
    Y = fb_tokens['Y']

    encoder_input_data = X
    decoder_data = Y
    # Shift the input and target for the decoder by one word
    decoder_input_data = np.array([i[:-1] for i in decoder_data], dtype=np.int32)
    decoder_target_data = np.array([i[1:] for i in decoder_data], dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(((encoder_input_data, decoder_input_data), decoder_target_data))

    # dataset is fairly small (c. 200mb) so use the cache
    dataset.cache()

    if flip_context:
        dataset = dataset.map(flip_context_func)

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size).batch(batch_size)

    # Use prefetch to allow model to get batches in the background while training
    dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def flip_context_func(input, target):
    '''Function for use with dataset.map() to reverse the context part
    '''
    encoder_input_data = input[0]
    flipped_encoder_input_data = encoder_input_data[::-1]
    return ((flipped_encoder_input_data, input[1]), target)


def load_from_txt(
    X_fpath, Y_fpath, word2vec_fpath, vocab_fpath, reverse_context=True, buffer_size=10000, batch_size=32
):
    def txt_generator():
        with open(X_fpath, 'r') as f:
            X = f.readlines()
        with open(Y_fpath, 'r') as f:
            Y = f.readlines()

        print('Loaded data from txt file')
        X = [x.strip('\n') for x in X]
        Y = [y.strip('\n') for y in Y]
        print('Removed new line characters')

        # Pipeline elements
        ws = WhiteSpaceTokenizer()
        padder = Padder('<pad>')

        pipe = Pipeline(steps=[('ws', ws), ('pad', padder)])
        X_processed = pipe.transform(X)
        Y_processed = pipe.transform(Y)
        print('ws tokenized and padded')

        for i, x in enumerate(X_processed):
            yield (x, Y_processed[i])

    # Word2Vec used to vectorise the encoder and decoder inputs
    word2vec = Word2Vec(special_vectors={'unknown': 0, '<sos>': 1})
    model = KeyedVectors.load_word2vec_format(word2vec_fpath)
    word2vec.set_model(model)
    print('Loaded word2vec mapping')

    inttk = IntegerTokenizer(vocab_fpath, add_to_vocab_if_not_present=False)

    ft = featurize(ftz=[word2vec.tf_map, inttk.tf_map])

    dataset = tf.data.Dataset.from_generator(
        txt_generator,
        output_types=(tf.string, tf.string),
        output_shapes=(tf.TensorShape((None,)), tf.TensorShape((None,))),
    )
    print('Created tf.data.Dataset')
    dataset = dataset.map(create_decoder_target)

    dataset = dataset.map(
        lambda x1, x2, y: tf.py_function(func=ft.tf_map, inp=[x1, x2, y], Tout=(tf.float32, tf.float32, tf.int32)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.map(lambda x1, x2, y: ((x1, x2), y))
    if reverse_context:
        dataset = dataset.map(flip_context_func)

    # dataset is fairly small (c. 200mb) so use the cache
    dataset.cache()

    # Shuffle and batch the dataset,
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    # Use prefetch to allow model to get batches in the background while training
    dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


class featurize():
    def __init__(self, ftz=[]):
        self.ftz = ftz

    def tf_map(self, x1, x2, y):
        x_1out = self.ftz[0](x1)
        x_2out = self.ftz[0](x2)
        y_out = self.ftz[1](y)
        return x_1out, x_2out, y_out


def create_decoder_target(x, y):
    return x, y[:-1], y[1:]
