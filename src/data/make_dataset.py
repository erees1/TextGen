# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import socialml
import os
import joblib
import numpy as np
import json
import msg_pipeline
from spellchecker import SpellChecker
from tqdm import tqdm
from word_utils import Vocab
import math
import yaml
from sklearn.pipeline import Pipeline

# Global variables
verbose = 1


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('interim_filepath', type=click.Path())
@click.argument('spec_filepath', type=click.Path())
@click.option('--test_split', default=0.0, type=float)
def main(input_filepath, output_filepath, interim_filepath, spec_filepath, test_split):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # parse specs from spec_filepath
    with open(spec_filepath) as f:
        specs = yaml.load(f, Loader=yaml.FullLoader)

    word2vec_path = os.path.join(input_filepath, specs['word2vec_binary'])

    socialml_dataset = extract_socialml_dataset(input_filepath, interim_filepath, specs)
    context_strings = socialml_dataset[0]
    response_strings = socialml_dataset[1]

    rs = msg_pipeline.RemoveCharsTransformer(specs['punc_list'])
    context_strings = rs.transform(context_strings)
    response_strings = rs.transform(response_strings)

    context_filepath = os.path.join(output_filepath, 'context_strings.txt')
    response_filepath = os.path.join(output_filepath, 'response_strings.txt')

    context_strings = [line + '\n' for line in context_strings]
    response_strings = [line + '\n' for line in response_strings]

    with open(context_filepath, 'w') as f:
        f.writelines(context_strings)
    with open(response_filepath, 'w') as f:
        f.writelines(response_strings)

    # context_vectors, response_tokens, vocab = training_data_pipeline(
    #     socialml_dataset, specs, word2vec_path, interim_filepath
    # )
    # vocab.save_vocab(os.path.join(output_filepath, 'vocab_pp2.json'))

    # if test_split > 0:
    #     X_train, X_test, Y_train, Y_test = train_test_split(context_vectors, response_tokens, test_split)

    #     train_dir = os.path.join(output_filepath, 'train')
    #     if not os.path.exists(train_dir):
    #         os.makedirs(train_dir)
    #     save_data(X_train, Y_train, os.path.join(train_dir, 'tokens'))

    #     test_dir = os.path.join(output_filepath, 'test')
    #     if not os.path.exists(test_dir):
    #         os.makedirs(test_dir)
    #     save_data(X_train, Y_train, os.path.join(test_dir, 'tokens'))
    # else:
    #     save_data(context_vectors, response_tokens, os.path.join(output_filepath, 'tokens'))


def extract_socialml_dataset(input_filepath, interim_filepath, specs):

    logger = logging.getLogger(__name__)

    # convert facebook json data into array
    fb_data_path = os.path.join(input_filepath, specs['fb_message_data'])
    if os.path.exists(fb_data_path):
        fb_message_array = socialml.FbMessenger(fb_data_path).extract(
            max_participants=specs['max_participants'],
            min_messages=2,
        )
        logger.info(f'extracted {len(fb_message_array)} conversations from fb archive')
    else:
        logger.info('No facebook database found')
        exit(1)

    # Convert imessage database into array
    imessage_path = os.path.join(input_filepath, specs['imessage_data'])
    if os.path.exists(imessage_path):
        imessage_array = socialml.IMessage(imessage_path
                                           ).extract(max_participants=specs['max_participants'], min_messages=2)
        logger.info(f'extracted {len(imessage_array)} conversations from imessage archive')
    else:
        logger.info('No imessage database found')
        exit(1)

    # Combine facebook and imessage data
    message_array = fb_message_array + imessage_array

    # Filter the arrays
    # Load the list of blocked words
    google_profanity_words_path = os.path.join(input_filepath, 'google_profanity_words.txt')
    with open(google_profanity_words_path, 'r') as f:
        google_profanity_words = f.readlines()
        # Remove end of line symbols
        google_profanity_words = [word.replace('\n', '') for word in google_profanity_words]

    message_array = socialml.filter_array(
        message_array,
        remove_hyperlinks=1,
        remove_words=(1, google_profanity_words),
        max_message_length=(1, specs['max_message_length']),
    )

    # Construct the database dictionary, this takes a LONG time if used
    if specs['correct_spelling']:
        logger.info('Correcting spelling in dataset')
        sc = msg_pipeline.SpellTransformer(os.path.join(interim_filepath, '40k_dictionary.json'))
        for c, conversation in enumerate(tqdm(message_array)):
            for m, message in enumerate(conversation):
                message_array[c][m] = sc.transform(message)

    # Convert the array to dataset pairs of (context, reponse)
    message_dataset = socialml.make_training_examples(
        message_array,
        max_context_length=specs['max_context'],
        combine_contexts=specs['combine_contexts'],
        add_seq_tags=False,
        verbose=1,
    )

    return message_dataset


def training_data_pipeline(message_dataset, specs, word2vec_path, interim_filepath):

    logger = logging.getLogger(__name__)

    contexts = message_dataset[0]  # 'X'
    responses = message_dataset[1]  # 'Y'

    # Pipeline elements
    special_vectors = {'unknown': 0}
    rs = msg_pipeline.RemoveCharsTransformer(specs['punc_list'])
    ws = msg_pipeline.WhiteSpaceTokenizer()
    tg = msg_pipeline.Tagger(
        specs['tags'][specs['tokens']['START_TOKEN']], specs['tags'][specs['tokens']['START_TOKEN']]
    )
    w2v = msg_pipeline.Word2Vec(word2vec_path, special_vectors)

    # vocab object for loading / saving dictionaries, starting only
    # from the tags so the vocab is built from scratch
    inttk = msg_pipeline.IntegerTokenizer(specs['tags'])

    context_pipe = Pipeline(steps=[('remove_chars', rs), ('ws_tokenizer', ws), ('word2vec', w2v)])
    responses_pipe = Pipeline(
        steps=[('remove_chars', rs), ('ws_tokenizer', ws), ('tagger', tg), ('int_tokenizer', inttk)]
    )

    context_vectors = context_pipe.transform(contexts)

    del context_pipe
    del w2v

    response_tokens = responses_pipe.transform(responses)

    # context_tokens, vocab = msg_pipeline.tokenize_and_pad(
    #     contexts,
    #     starting_vocab=vocab,
    #     add_if_not_present=True,
    #     verbose=verbose,
    # )
    # response_tokens, vocab = msg_pipeline.tokenize_and_pad(
    #     responses,
    #     starting_vocab=vocab,
    #     add_if_not_present=True,
    #     verbose=verbose,
    # )
    vocab = inttk.vocab

    logger.info(f'Created and tokenized dataset with {len(response_tokens)} examples')

    return context_vectors, response_tokens, vocab


def train_test_split(X, Y, test_split):

    if test_split > 0:
        n_examples = len(X)
        n_test = int(test_split * n_examples)
        choice = np.random.choice(range(n_examples), size=n_test, replace=False)
        test_idx = np.zeros(n_examples, dtype=bool)
        test_idx[choice] = True
        train_idx = ~test_idx

        X_train = X[train_idx]
        Y_train = Y[train_idx]
        X_test = X[test_idx]
        Y_test = Y[test_idx]

    return X_train, X_test, Y_train, Y_test

def save_data(X, Y, output_filepath):
    if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        np.savez(output_filepath, X=X, Y=Y)
    elif isinstance(X, list) and isinstance(Y, list):
        joblib.dump(X, os.path.join(output_filepath + '_X.gz'))
        joblib.dump(X, os.path.join(output_filepath + '_Y.gz'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
