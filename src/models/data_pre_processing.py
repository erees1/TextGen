# -*- coding: utf-8 -*-
import socialml
import numpy as np
import tensorflow as tf

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

    dataset = tf.data.Dataset.from_tensor_slices(
        ((encoder_input_data, decoder_input_data), decoder_target_data)
    )

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
