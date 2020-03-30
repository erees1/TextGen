# -*- coding: utf-8 -*-
from pathlib import Path
import json
import os
import numpy as np
import sys
from tqdm import tqdm


def create_vocab_file_from_dictionary(dictionary, tags, save_dir):
    """Create vocab file (list of words) from word frequency dictionary, prepending reqiured tags

    Arguments:
        dictionary {str or dict} -- Dictionary of word frequencys or path to json object
        tags {str or list} -- List of tags or path to json object, will be prepended to vocab file
        save_dir {str} -- path to save vocab file
    """
    if isinstance(dictionary, str):
        # save dictionary file words along with word counts
        with open(os.path.join(save_dir, dictionary), 'r') as f:
            dictionary = json.load(dictionary, f)

    if isinstance(tags, str):
        # save dictionary file words along with word counts
        with open(os.path.join(save_dir, tags), 'r') as f:
            tags = json.load(tags, f)

    vocab = [keys for keys, _ in sorted(dictionary.items(), key=lambda pair: pair[1])]
    vocab = tags + vocab

    # vocab file is a list only (unlike dictionary that has counts)
    with open(os.path.join(save_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)

class Vocab():
    def __init__(self, vocab, unknown_token='<unk>'):
        if vocab is None:
            self.int2word = []
        else:
            with open(vocab) as f:
                self.int2word = list(json.load(f))

        # int2word and word2int map words to and from integers
        self.int2word = np.array(self.int2word)
        self._create_word2int()
        # user_added tracks words added to the ditionary that weren't in the intial one
        self.user_added = {}
        self.unknown_token = unknown_token

    def __len__(self):
        return len(self.int2word)

    def add_word(self, word):
        if word not in self.word2int:
            self.user_added[word] = 1
            # Add word to int2word and user_added
            self._add_to_int2word(word)
            self._add_to_word2int(word, len(self.int2word) - 1)
            assert len(self.word2int) == len(self.int2word)
        if word in self.user_added:
            # for user added words the total is added to the dictionary
            self.user_added[word] += 1

    def add_words(self, words):
        for word in words:
            self.add_word(word)
        return

    def save_vocab(self, fname):
        with open(fname, 'w') as f:
            json.dump(list(self.int2word), f)

    def convert2int(self, ls, add_if_not_present=False):
        '''
        Takes a list and maps it to integers
        '''
        out = []
        if not (isinstance(ls, list) or isinstance(ls, np.ndarray)):
            raise Exception('Must provide a list or array of words')
        for word in ls:
            try:
                out.append(self.word2int[word])
            except KeyError:
                if add_if_not_present:
                    self.add_word(word)
                    int_ = self.word2int[word]
                else:
                    int_ = self.word2int[self.unknown_token]
                out.append(int_)
        return out

    def convert2word(self, ls):
        '''
        Takes a list of ints and converts to words
        '''
        if not (isinstance(ls, list) or isinstance(ls, np.ndarray)):
            raise Exception('Must provide a list or array of words')
        return [self.int2word[i] for i in ls]

    def _create_word2int(self):
        self.word2int = {e: i for i, e in enumerate(self.int2word)}

    def _add_to_int2word(self, word):
        self.int2word = np.hstack((self.int2word, word))

    def _add_to_word2int(self, word, index):
        self.word2int[word] = index
