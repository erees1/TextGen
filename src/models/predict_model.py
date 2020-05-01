# -*- coding: utf-8 -*-
import os
from src.data.word_utils import Vocab
from src.data.msg_pipeline import IntegerTokenizer, SpellTransformer, RemoveCharsTransformer, WhiteSpaceTokenizer, Word2Vec, Tagger
from gensim import KeyedVectors
from sklearn.pipeline import Pipeline
import numpy as np
import tensorflow as tf
from src.models import seq2seq
from itertools import chain
import yaml
import logging
import re


def latest_checkpoint(checkpoint_path):
    """Get filepath of latest model in checkpoint directory"""
    files = [os.path.join(checkpoint_path, file) for file in os.listdir(checkpoint_path)]
    files = [file for file in files if '.json' not in file]
    modified = [os.path.getmtime(file) for file in files]
    files = [file for file, _ in sorted(zip(files, modified), key=lambda pair: pair[1], reverse=True)]
    return files[0]


def load_yaml(filepath):
    with open(filepath) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_encoder_state(encoder_model, input_as_int):
    """Expand the dimensions so we have a batch size of 1"""
    input_as_int = tf.expand_dims(input_as_int, axis=0)
    states_value = encoder_model.predict(input_as_int)
    return states_value


def argmax_select(arr):
    """Select argmax from  an arra"""
    idx = np.argmax(arr)
    p = arr[idx]
    return idx, p


def get_top_n(arr, n):
    """Return largest n indices and values from array"""
    if tf.is_tensor(arr):
        arr = arr.numpy()
    flat = arr.flatten()
    idxs = np.argpartition(flat, -n)[-n:]
    idxs = idxs[np.argsort(-flat[idxs])]
    ps = flat[idxs]
    idxs = np.unravel_index(idxs, arr.shape)
    # idxs is a tuple with first element
    # idxs = [tuple(idx) for idx in np.swapaxes(idxs, 0, 1)]
    return idxs, ps


class SeqInference():
    """Handles the inference loop for a single message"""
    def __init__(
        self,
        vocab_filepath,
        encoder_model,
        decoder_model,
        model_spec_file,
        data_spec_file,
        method='arg_max',
        beam_width=3,
        dictionary_dir=None,
        max_decoder_seq_length=28,
        verbose=0,
        pipeline='word2vec',
        word2vec_fpath='',
        **kwargs,
    ):

        self.data_spec_file = load_yaml(data_spec_file)
        self.model_spec_file = load_yaml(model_spec_file)
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.method = method
        self.beam_width = beam_width
        self.max_decoder_seq_length = max_decoder_seq_length
        self.dictionary_dir = dictionary_dir
        self.verbose = verbose
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        self.logger = logging.getLogger(__name__)

        self._get_tags_from_spec()

        # Pipeline Elements
        self.int_tokenizer = IntegerTokenizer(vocab_filepath)
        rs = RemoveCharsTransformer(self.data_spec_file['punc_list'])
        ws = WhiteSpaceTokenizer()
        self.tg = Tagger()

        self.available_pipelines = ['integertokenizer', 'word2vec']
        self.pipe = Pipeline(steps=[('remove_chars', rs), ('white_space_tokenizer', ws)])

        if self.dictionary_dir is not None:
            self.pipe.append(('spelling', st))

        # self.pipe.append(('tagger', tg))

        if pipeline == 'integertokenizer':
            self.pipe.append(('integer_tokenizer', self.int_tokenizer))

        elif pipeline == 'word2vec':
            word2vec = Word2Vec()
            word2vec.set_model(KeyedVectors.load_word2vec_format(word2vec_fpath))
            self.pipe.append(('integer_tokenizer', self.word2vec))

        else:
            raise KeyError(f'Unavailablve pipeline specified, please choose one of {self.available_pipelines}')

    def _log(self, message):
        if self.verbose > 1:
            self.logger.info(message)
        elif self.verbose == 1:
            print(message)

    def _get_tags_from_spec(self):
        self.START_TOKEN = self.data_spec_file['tokens']['START_TOKEN']
        self.END_TOKEN = self.data_spec_file['tokens']['END_TOKEN']
        self.START_TAG = self.data_spec_file['tags'][self.START_TOKEN]
        self.END_TAG = self.data_spec_file['tags'][self.END_TOKEN]

    def predict_response_from_text(self, message):
        reverse = self.model_spec_file['tf_dataset_params']['reverse_context']
        self._log(f'Tokenizing mesage: {message}')
        input_vectors = np.squeeze(self.process_message([message], reverse))
        output_tokens = self._predict_response_from_tokens(input_vectors)
        response = " ".join(self.int_tokenizer.inverse_transform(output_tokens))
        response = np.squeeze(self.tg.inverse_transform([response]))
        return response

    def process_message(self, input_string, reverse):
        """turn string into a tokenized array"""

        # Settings
        input_as_int = self.pipe.transform(input_string)
        if reverse:
            input_as_int = input_as_int[::-1]
        return np.asarray(input_as_int)

    def strip_tags_from_text(self, message):
        tag_match = re.compile('<[a-z]{3}>')
        message = tag_match.sub('', message).strip()
        message = re.sub(' +', ' ', message)
        return message

    def _predict_response_from_tokens(self, context):

        self._log(f'Sending context: {context} to encoder')
        # Encoded the context (messages up to this point)
        encoder_states = get_encoder_state(self.encoder_model, context)
        # Inital value of target sequence is the start sequence tag
        start_token = np.array(self.START_TOKEN)
        # First state for the decoder is the encoder states
        states = encoder_states
        if self.method == 'arg_max':
            decoded_tokens = self._argmax_loop(start_token, encoder_states)
        elif self.method == 'beam_search':
            decoded_tokens = self._beam_search_loop(start_token, encoder_states)

        return decoded_tokens

    def _argmax_loop(self, target_seq, states):
        stop_condition = False
        decoded_tokens = []

        while not stop_condition:
            output_tokens, states = self._predict_next_char(target_seq, states)

            # Collapse the output tokens to a single vector
            probs = tf.squeeze(output_tokens)
            sampled_index, p = argmax_select(probs)
            sampled_word = self.int_tokenizer.inverse_transform(sampled_index)
            sampled_vector = self.pipe.transform(sampled_word)

            target_seq = tf.convert_to_tensor([[sampled_vector]])

            # Exit condition: either hit max length or find stop character.
            if (sampled_index == self.END_TAG or len(decoded_tokens) > self.max_decoder_seq_length):
                stop_condition = True
            else:
                decoded_tokens.append(sampled_index)

        return decoded_tokens

    def _beam_search_loop(self, inital_target_seq, encoder_states):
        beam_width = self.beam_width
        stop_condition = False
        decoded_tokens = []
        vocab_length = len(self.tokenizer.vocab)

        # prob beam sequence keeps track of P(seq)for each beam and is updated at each iter
        log_prob_beam_seq = np.log(np.ones((beam_width, 1)))
        # prob_char_given_prev tracks conditional probability of all characters given the previous
        log_prob_char_given_prev = np.empty((beam_width, vocab_length))

        beam_seq = np.empty((beam_width, 1), dtype=np.int32)
        beam_seq[:, 0] = [inital_target_seq] * 3
        beam_has_ended = [False] * beam_width
        final_tokens = []
        final_log_probs = []

        first_char = True
        stop_condition = False

        while not stop_condition:

            if first_char:
                decoder_output, states_values = self._predict_next_char(np.asarray([inital_target_seq]), encoder_states)
                log_prob_char_given_prev[0] = np.log(tf.squeeze(decoder_output).numpy())
                beam_states = [states_values] * beam_width

            else:
                for beam in range(beam_width):
                    if not beam_has_ended[beam]:
                        # Last character of the beam sequence is the input for the decoder
                        # Convert beam_seq which has integer words into vectors
                        prev_word = self.int_tokenizer.inverse_transform(beam_seq[beam][-1])
                        prev_word_as_vectors = self.pipe.transform(prev_word)

                        decoder_input = np.asarray([prev_word_as_vectors])
                        decoder_output, states_values = self._predict_next_char(decoder_input, beam_states[beam])
                        log_prob_char_given_prev[beam] = np.log(tf.squeeze(decoder_output).numpy())
                        beam_states[beam] = states_values
                    else:
                        log_prob_char_given_prev[beam] = [-np.inf] * vocab_length

            # Probability of all characters
            if first_char:
                log_prob_seq = log_prob_char_given_prev[0] + log_prob_beam_seq[0]
                log_prob_seq = log_prob_seq.reshape((1, -1))
            else:
                log_prob_seq = log_prob_char_given_prev + log_prob_beam_seq
                assert log_prob_seq.shape == (beam_width, vocab_length)

            # Carry forward the top beam_width
            top_n, log_p = get_top_n(log_prob_seq, beam_width)
            log_prob_beam_seq = log_p.reshape((-1, 1))

            # Add top_n to the beam_seq
            beam_states = [beam_states[i] for i in top_n[0]]
            beam_seq[:, :] = [beam_seq[:, :][i] for i in top_n[0]]
            beam_seq = np.hstack((beam_seq, top_n[1].reshape(beam_width, 1)))
            self._log(f'Current beam sequence \n {beam_seq}')

            for beam in range(beam_width):
                if beam_seq[beam, -1] == self.END_TOKEN or len(beam_seq[beam, :]) >= self.max_decoder_seq_length:
                    beam_has_ended[beam] = True
                    self._log(f'Appending {beam_seq[beam]} to final tokens')
                    if len(beam_seq[beam]) > 3:
                        final_tokens.append(np.array(beam_seq[beam]))
                        final_log_probs.append(np.array(log_prob_beam_seq[beam]))
                    if len(final_tokens) >= beam_width:
                        stop_condition = True

            first_char = False
            # End of while loop

        return final_tokens[np.argmax(final_log_probs)]

    def _predict_next_char(self, target_seq, states_value):
        """Take a single input and lstm state and predict the next item and state"""
        output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
        states_value = [h, c]
        return output_tokens, states_value


class Chat(SeqInference):
    def __init__(
        self,
        vocab_filepath,
        encoder_model,
        decoder_model,
        model_spec_file,
        data_spec_file,
        method='arg_max',
        beam_width=3,
        dictionary_dir=None,
        max_decoder_seq_length=28,
        verbose=0,
        max_context=7,
    ):
        super().__init__(
            vocab_filepath,
            encoder_model,
            decoder_model,
            model_spec_file,
            data_spec_file,
            method=method,
            beam_width=beam_width,
            dictionary_dir=dictionary_dir,
            max_decoder_seq_length=max_decoder_seq_length,
            verbose=verbose,
        )
        self.context = []
        self.history = []
        self.max_context = max_context
        self.reverse_context = self.model_spec_file['tf_dataset_params']['reverse_context']

    def send(self, message):
        # Add message to history
        self.history.append(message)
        # Convert message from words into tokens (ints)

        # Don't reverse the message here as it is reversed when added to context
        message_as_tokens = self.process_message(message, reverse=False)
        self.add_tokens_to_context(message_as_tokens)
        flattend_context = np.concatenate(self.context)
        output_tokens = self._predict_response_from_tokens(flattend_context)
        response = " ".join(self.tokenizer.inverse_transform(output_tokens))
        response = self.strip_tags_from_text(response)
        self.add_tokens_to_context(output_tokens)
        self.history.append(response)

        return response

    def add_tokens_to_context(self, tokens):
        if len(self.context) >= self.max_context:
            pop_old = True
        else:
            pop_old = False
        if self.reverse_context:
            if pop_old:
                self.context.pop(0)
            tokens = tokens[::-1]
            context = self.context.insert(0, tokens)
        else:
            if pop_old:
                self.context.pop()
            self.context = self.context.append(tokens)

    def reset_context(self):
        self.context = []
