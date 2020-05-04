# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import Model, layers
import tensorflow_hub as hub
import numpy as np


class PreTrainedEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_list, mask_value=0, name='embed', **kwargs):
        super(PreTrainedEmbedding, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.pretrained_embeddings = self._pretrained_embeddings(vocab_list)
        self.embedding = tf.nn.embedding_lookup

    def call(self, inputs):
        x = self.embedding(self.pretrained_embeddings, inputs)
        return x

    def compute_mask(self, inputs, mask=None):
        return inputs == self.mask_value

    def _pretrained_embeddings(self, vocab_list):
        embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50-with-normalization/1")
        return self._build_embeddings_from_vocab(vocab_list, embed)

    def _build_embeddings_from_vocab(self, vocab_list, embedding_func, n_oov=1, insert_pad=True):
        # Insert 0 for padding
        vocab_list = vocab_list[:]
        if insert_pad:
            vocab_list.insert(0, '')
        embedding_length = embedding_func(['hello']).shape[1]
        output = np.zeros((len(vocab_list) + n_oov, embedding_length), dtype=np.float32)
        output[:-n_oov][:] = embedding_func(vocab_list).numpy()
        return output


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""
    def __init__(self, n_units=64, depth=2, dropout=0.0, name='encoder', **kwargs):
        super().__init__(**kwargs)
        self.lstms = [
            layers.LSTM(
                n_units,
                return_sequences=True,
                return_state=True,
                dropout=dropout,
                name=f'en_lstm-{layer + 1}',
            ) for layer in range(depth)
        ]

    def call(self, x):
        states = []
        for lstm in self.lstms:
            x, state_h, state_c = lstm(x)
            states.append([state_h, state_c])
        return states


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""
    def __init__(self, vocab_size, n_units=64, depth=2, dropout=0.0, name='decoder', **kwargs):
        super().__init__(**kwargs)
        self.lstms = [
            layers.LSTM(
                n_units,
                return_sequences=True,
                return_state=True,
                dropout=dropout,
                name=f'de_lstm-{layer + 1}',
            ) for layer in range(depth)
        ]
        self.dense = layers.TimeDistributed(layers.Dense(vocab_size, name='decoder_ouput'))

    def call(self, x, encoder_states):
        states = []
        for i, lstm in enumerate(self.lstms):
            x, state_h, state_c = lstm(x, initial_state=encoder_states[i])
            states.append([state_h, state_c])

        output = self.dense(x)
        return output, states


class LSTMSeqModel(Model):
    """Combines the encoder and decoder into an end-to-end model for training."""
    def __init__(self, vocab_list, depth=2, n_units=64, name='lstm_seq2seq', **kwargs):

        super().__init__(**kwargs)
        with tf.name_scope('inputs'):
            self.embeddings = PreTrainedEmbedding(vocab_list, mask_value=0)
        with tf.name_scope('encoder'):
            self.encoder = Encoder(n_units=n_units, depth=depth)
        with tf.name_scope('decoder'):
            self.decoder = Decoder(len(vocab_list), n_units=n_units, depth=depth)

    def call(self, inp, tar_inp):

        encoder_inputs = self.embeddings(inp)
        decoder_inputs = self.embeddings(tar_inp)
        encoder_states = self.encoder(encoder_inputs)
        output, states = self.decoder(decoder_inputs, encoder_states)
        return output, states
