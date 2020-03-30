# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

def build_model(encoder_depth,
                encoder_units,
                decoder_depth,
                embedding_dim,
                vocab_size):

    # Embedding layer is shared by both encoder and decoder inputs to limit parameters
    input_embedding = Embedding(vocab_size, embedding_dim, name='encoder_embedding', mask_zero=True)

    # Input is an embedded vector
    encoder_inputs = Input(shape=(None,), name='encoder_input')
    x = input_embedding(encoder_inputs)
    i = 0
    encoder_outputs, state_h, state_c = LSTM(encoder_units, return_state=True, name=f'encoder_lstm-{i+1}')(x)
    # Only care about the states of the encoder
    encoder_states = [state_h, state_c]

    # Decoder recieves inputs and the state of the encoder
    decoder_inputs = Input(shape=(None,), name='decoder_input')
    embedded_decoder_inputs = input_embedding(decoder_inputs)
    decoder_lstm = LSTM(
        encoder_units,
        return_sequences=True,
        return_state=True,
        name=f'decoder_lstm-{i+1}',
    )
    decoder_ouput, _, _ = decoder_lstm(embedded_decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_ouput')
    decoder_outputs = decoder_dense(decoder_ouput)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Inference model
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(encoder_units,))
    decoder_state_input_c = Input(shape=(encoder_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(embedded_decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

def compile_model(model):
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')
    return model

def build_inference_model(model):
    encoder_inputs = model.get_layer('encoder_input')
    encoder_lstm = model.get_layer('encoder_lstm-1')
    encoder_output = model.get_layer('encoder_states')
