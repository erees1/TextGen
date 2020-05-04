# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding


class LSTMSeqModel(Model):
    def __init__(self):
        pass

    def build_model(
        self,
        depth,
        encoder_units,
        vocab_size,
        use_embedding_layer=False,
        embedding_dim=100,
        encoder_input_shape=(300),
        decoder_input_shape=(300),
        batch_size=None,
    ):

        # BUILD ENCODER FOR TRAINING
        self.encoder_inputs = Input(shape=encoder_input_shape, batch_size=batch_size, name='encoder_input')
        embedding

        encoder_outputs = []
        self.encoder_states = []
        x_encoder = embeded_encoder_inputs
        for layer in range(depth):
            encoder_layer_outputs, state_h, state_c = LSTM(
                encoder_units,
                return_sequences=True,
                return_state=True,
                name=f'encoder_lstm-{layer + 1}',
            )(x_encoder)
            x_encoder = encoder_layer_outputs
            encoder_outputs.append(encoder_layer_outputs)
            # Only care about the states of the encoder
            self.encoder_states.append([state_h, state_c])

        # BUILD DECODER FOR TRAINING
        decoder_inputs = Input(shape=decoder_input_shape, batch_size=batch_size, name='decoder_input')
        if use_embedding_layer:
            embedded_decoder_inputs = input_embedding(decoder_inputs)
        else:
            embedded_decoder_inputs = decoder_inputs

        decoder_layers = []
        x_decoder = embedded_decoder_inputs
        for layer in range(depth):
            decoder_layer = LSTM(
                encoder_units,
                return_sequences=True,
                return_state=True,
                name=f'decoder_lstm-{layer + 1}',
            )
            decoder_layers.append(decoder_layer)
            x_decoder, _, _ = decoder_layer(x_decoder, initial_state=self.encoder_states[layer])
        decoder_output = x_decoder

        decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_ouput')
        decoder_outputs = decoder_dense(x_decoder)
        self.model = Model([self.encoder_inputs, decoder_inputs], decoder_output)

        # Inference model --------------------------------------------------------

        # self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

        # decoder_states_inputs = []
        # decoder_states_outputs = []
        # x_decoder = embedded_decoder_inputs
        # for layer in range(depth):
        #     decoder_layer_state_input_h = Input(shape=(encoder_units, ), batch_size=batch_size, name=f'decoder_h_input-{layer+1}')
        #     decoder_layer_state_input_c = Input(shape=(encoder_units, ), batch_size=batch_size, name=f'decoder_c_input-{layer+1}')
        #     decoder_layer_state_inputs = [decoder_layer_state_input_h, decoder_layer_state_input_c]
        #     decoder_states_inputs.append(decoder_layer_state_inputs)

        #     decoder_outputs, state_h, state_c = decoder_layers[layer](
        #         x_decoder, initial_state=decoder_layer_state_inputs
        #     )
        #     x_decoder = decoder_outputs
        #     decoder_states_outputs.append([state_h, state_c])

        # decoder_outputs = decoder_dense(decoder_outputs)
        # self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states_outputs)

        # self.inferenece_model = [self.encoder_model, self.decoder_model]
        # , encoder_model, decoder_model

    def compile_model(self):
        self.model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')


class PreTrainedEmbedding(tf.keras.layers.Layer):
    def __init__(self, mask_value='', **kwargs):
        super(PreTrainedEmbedding, self).__init__(**kwargs)
        self.embedding = hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50-with-normalization/1", dtype=tf.string,
            input_shape=[[]], output_shape=[50]
        )
        self.mask_value = mask_value

    def call(self, inputs):
        x = self.embedding(inputs)
        return x

    def compute_mask(self, inputs, mask=None):
        return inputs == self.mask_value

