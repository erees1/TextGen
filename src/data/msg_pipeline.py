# -*- coding: utf-8 -*-
import numpy as np
import socialml
from src.data.word_utils import Vocab
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

from gensim.models import KeyedVectors
import re
from typing import List

# Pipeline transformers (compaitble with sklearn pipe)

class Word2Vec(BaseEstimator, TransformerMixin):
    def __init__(self, special_vectors={'<unk>': 0, '<pad>': 0}):
        self.special_vectors = special_vectors

    def set_model(self, model):
        self.model = model
        self.vector_dim = len(self.model['hello'])

    def fit(self, X):
        return self

    def transform(self, X):
        X_new = [None] * len(X)
        for i, example in enumerate(tqdm(X)):
            vectors = np.empty((len(example), self.vector_dim, ))
            for w, word in enumerate(example):
                vectors[w, :] = self[word]
            X_new[i] = vectors
        return X_new

    def map(self, X):
        vectors = np.empty((len(X), self.vector_dim))
        for w, x in enumerate(X):
            vectors[w, :] = self[x]
        return vectors

    def tf_map(self, X):
        vectors = np.empty((len(X), self.vector_dim))
        for w, x in enumerate(X.numpy()):
            vectors[w, :] = self[x.decode()]
        return vectors

    def __getitem__(self, index):
        vector = np.empty((self.vector_dim,))
        try:
            vector[:] = [self.special_vectors[index]] * self.vector_dim
        except KeyError:
            try:
                vector[:] = self.model[index]             
            except KeyError:
                vector[:] = [self.special_vectors['unknown']] * self.vector_dim
        return vector


class Padder(BaseEstimator, TransformerMixin):
    def __init__(self, value, padding='post'):
        self.value = value
        self.padding = padding

    def fit(self, X):
        return self

    def transform(self, X):
        max_length = max([len(row) for row in X])
        new_X = [None] * len(X)
        for i, row in enumerate(X):
            if self.padding == 'post':
                new_X[i] = row + [self.value] * (max_length - len(row))
            elif self.padding == 'pre':
                new_X[i] = [self.value] * (max_length - len(row)) + row
        return new_X


class Tagger(BaseEstimator, TransformerMixin):
    def __init__(self, start_tag, end_tag):
        self.start_tag = start_tag
        self.end_tag = end_tag

    def fit(self, X):
        return self

    def transform(self, X):
        if isinstance(X[0], str):
            return [' '.join([self.start_tag, x, self.end_tag]) for x in X]
        else:
            isinstance(X[0], list)
        return [[self.start_tag] + i + [self.end_tag] for i in X]

    def inverse_transform(self, X):
        tag_match = re.compile('<[a-z]{3}>')
        X_out = [tag_match.sub('', message).strip() for message in X]
        X_out = [re.sub(' +', ' ', message) for message in X_out]
        return X_out


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

    def tf_map(self, x):
        x_decode = [w.decode() for w in x.numpy()]
        x_new = self.vocab.convert2int(x_decode, add_if_not_present=self.add_to_vocab_if_not_present)
        return x_new

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

    def fit(self, X):
        return self

    def transform(self, X):
        if isinstance(X, str):
            X = self._clean_text(X)
        else:
            for i, seq in enumerate(X):
                X[i] = self._clean_text(seq)

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
        from spellchecker import SpellChecker
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

def tensorflow_data_pipeline():
    pass