# -*- coding: utf-8 -*-
import click
import logging
import os
from src.data.word_utils import Vocab
from src.data.msg_pipeline import Word2Vec
from gensim.models import KeyedVectors

@click.command()
@click.argument('kv_filepath', type=str)
@click.argument('vocab_filepath', type=str)
@click.argument('output_filepath', type=str)
def main(kv_filepath, vocab_filepath, output_filepath):

    model = KeyedVectors.load_word2vec_format(kv_filepath, binary=True)
    vocab = Vocab(vocab_filepath)
    short_kv = KeyedVectors(vector_size=len(model['hello']))

    for word in vocab.word2int.keys():
        try:
            short_kv.add(word, model[word])
        except KeyError:
            continue

    short_kv.save_word2vec_format(os.path.join(output_filepath, 'short-vectors.bin'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
