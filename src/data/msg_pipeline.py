# -*- coding: utf-8 -*-
import numpy as np
import socialml
from src.data.word_utils import Vocab
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from spellchecker import SpellChecker
from gensim.models import KeyedVectors
import re
from typing import List

def tokenize_and_pad(dataset, starting_vocab=None, add_if_not_present=True, verbose=0, tqdm_unit_scale=False):
    '''Pad and tokenize conversations

    Args:
        vocab (int): maximum number of particpants (optional)
        add_if_not_present (bool): Whether to add word to vocab or use <unk> token

    Returns:
        tokens_array: 2d numpy array of zero padded tokenized conversations
    '''
    # Load starting vocab if not None
    if starting_vocab is not None:
        if not add_if_not_present:
            raise Exception('Must add if not present if no starting vocab')
        if isinstance(starting_vocab, str):
            vocab = Vocab(starting_vocab)
        else:
            vocab = starting_vocab
    else:
        vocab = Vocab(vocab=None)

    # predefine an array of the correct size
    list_of_tokens = []
    max_length = max([len(i.split(' ')) for i in dataset])
    tokens_array = np.zeros((len(dataset), max_length), dtype=np.int32)

    # Populate the array
    if verbose < 1:
        disable_tqdm = True
    else:
        disable_tqdm = False
    for i, example in enumerate(tqdm(dataset, 'tokenizing dataset', disable=disable_tqdm, unit_scale=tqdm_unit_scale)):
        # Convert words to ints and add to the array
        words = example.lower().split(' ')
        tokens = vocab.convert2int(words, add_if_not_present)
        tokens = np.asarray(tokens, dtype=np.int32)
        tokens_array[i, :len(tokens)] = tokens

        # Just for printing progress
        # print_progress(i, len(dataset))

    return tokens_array, vocab

# Pipeline transformers (compaitble with sklearn pipe)

class Word2Vec(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec_path, special_vectors):
        self.model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        self.vector_dim = len(self.model['hello'])
        self.special_vectors = special_vectors

    def fit(self, X):
        return self

    def transform(self, X):
        X_new = [None] * len(X)
        for i, example in enumerate(tqdm(X)):
            vectors = np.empty((len(example), self.vector_dim, ))
            for w, word in enumerate(example):
                try:
                    vectors[w, :] = self.model[word]
                except KeyError:
                    vectors[w, :] = self.special_vectors['unknown']
            X_new[i] = vectors
        return X_new


class Padder(BaseEstimator, TransformerMixin):
    def __init__(self, padding):
        self.padding = padding

    def transform(self, X):
        max_length = max([len(row) for row in X])
        new_X = [[''] * max_length] * len(X)
        for i, row in enumerate(X):
            new_X[i][(max_length - len(row)):]
        return new_X


class Tagger(BaseEstimator, TransformerMixin):
    def __init__(self, start_tag, end_tag):
        self.start_tag = start_tag
        self.end_tag = end_tag

    def fit(self, X):
        return self

    def transform(self, X):
        return [[self.start_tag] + i + [self.end_tag] for i in X]


class WhiteSpaceTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return X

    def transform(self, X: List[str]):
        X_new = [None] * len(X)
        for i, example in enumerate(X):
            X_new[i] = example.split(' ')
        return X_new

    def inverse_transform(self, X):
        X_new = [None] * len(X)
        for i, example in enumerate(X):
            X_new[i] = example.split(' ')
        return X_new


class IntegerTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, vocab_path, add_to_vocab_if_not_present=True):
        self.vocab = Vocab(vocab_path)
        self.add_to_vocab_if_not_present = add_to_vocab_if_not_present

    def fit(self, X):
        return self

    def transform(self, X):
        X_new = [None] * len(X)
        for i, example in enumerate(X):            
            X_new[i] = self.vocab.convert2int(example, add_if_not_present=self.add_to_vocab_if_not_present)
        return X_new

    def inverse_transform(self, X):
        if not (isinstance(X, list) or isinstance(X, np.ndarray)):
            X = [X]
            single_input = True
        else:
            single_input = False
        X = self.vocab.convert2word(X)
        if single_input:
            X = X[0]
        return X

class RemoveCharsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, chars):
        self.chars = chars

    def fit(self):
        return self

    def transform(self, X):
        if isinstance(X, str):
            X = self._clean_text(X)
        else:
            for i, seq in enumerate(X):
                X[i] = self._clean_text(seq)
                if X[i] == '':
                    del X[i]
        return X

    def _clean_text(self, text):
        if isinstance(self.chars, list):
            punc_list = self.chars
            t = str.maketrans(dict.fromkeys(punc_list, ''))
            text = text.translate(t).lower()
            text = text.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
        elif self.chars == 'all':
            regex = re.compile('<[a-z]{3}>|[a-zA-Z ]')
            text = regex.findall(text.lower())
            text = ''.join(text).strip()
            text = re.sub(' +', ' ', text)
        return text

class SpellTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dictionary_path):
        self.spell = SpellChecker(local_dictionary=dictionary_path)

    def fit(self):
        return self

    def transform(self, X):
        if isinstance(X, str):
            X = correct_text(X, self.spell)
        else:
            X = correct_words(X, self.spell)
        return X

def correct_words(words, spell):
    """Correct spelling of a list or words using spellcheker object

    Arguments:
        words {list} -- list of words to correct
        spell {SpellChecker Object} -- Ojbect with which to carry out correction

    Returns:
        words -- list of corrected words
    """
    words = [word.lower() for word in words]
    unknown = spell.unknown(words)
    for i, word in enumerate(words):
        if word.lower() in unknown:
            words[i] = spell.correction(word)
    return words

def correct_text(text, spell):
    '''Correct spelling of a string using the pyspellchecker module
    '''
    words = text.lower().split(' ')
    unknown = spell.unknown(words)
    for i, word in enumerate(words):
        if word in unknown:
            words[i] = spell.correction(word)

    return ' '.join(words)
