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

    MAX_PARTICIPANTS = int(specs['max_participants'])
    MAX_CONTEXT = int(specs['max_context'])
    MAX_MESSAGE_LENGTH = int(specs['max_message_length'])
    REMOVE_HYPERLINKS = specs['remove_hyperlinks']
    ADD_SEQ_TAGS = specs['add_seq_tags']
    COMBINE_CONTEXTS = specs['combine_contexts']
    CORRECT_SPELLING = specs['correct_spelling']
    punc_list = specs['punc_list']

    verbose = 1

    # START actual data processing

    # convert facebook json data into array
    fb_data_path = os.path.join(input_filepath, 'fb_messages/inbox')
    if os.path.exists(fb_data_path):
        fb_message_array = socialml.FbMessenger(fb_data_path).extract(max_participants=MAX_PARTICIPANTS, min_messages=2)
        logger.info(f'extracted {len(fb_message_array)} conversations from fb archive')
    else:
        logger.info('No facebook database found')
        exit(1)

    # Convert imessage database into array
    imessage_path = os.path.join(input_filepath, 'imessage/chat.db')
    if os.path.exists(imessage_path):
        imessage_array = socialml.IMessage(imessage_path).extract(max_participants=MAX_PARTICIPANTS, min_messages=2)
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
        message_array, remove_hyperlinks=1, remove_words=(1, google_profanity_words),
        max_message_length=(1, MAX_MESSAGE_LENGTH)
    )

    # # Construct the database dictionary, this takes a LONG time if used
    if CORRECT_SPELLING:
        logger.info('Correcting spelling in dataset')
        sc = msg_pipeline.SpellTransformer(os.path.join(interim_filepath, '40k_dictionary.json'))
        for c, conversation in enumerate(tqdm(message_array)):
            for m, message in enumerate(conversation):
                message_array[c][m] = sc.transform(message)

    # Convert the array to dataset pairs of (context, reponse)
    message_dataset = socialml.make_training_examples(
        message_array,
        max_context_length=MAX_CONTEXT,
        combine_contexts=COMBINE_CONTEXTS,
        add_seq_tags=ADD_SEQ_TAGS,
        verbose=verbose,
    )

    # Func to remove symbols, numbers and tokenize the text
    rs = msg_pipeline.RemoveCharsTransformer(punc_list)

    # Use the clean_text function to remove punctuation
    contexts = message_dataset[0]
    responses = message_dataset[1]
    contexts = rs.transform(contexts)
    responses = rs.transform(responses)

    # vocab object for loading / saving dictionaries, starting only
    # from the tags so the vocab is built from scratch
    vocab = Vocab(os.path.join(interim_filepath, 'tags.json'))

    context_tokens, vocab = msg_pipeline.tokenize_and_pad(
        contexts, starting_vocab=vocab, add_if_not_present=True, verbose=verbose
    )
    response_tokens, vocab = msg_pipeline.tokenize_and_pad(
        responses, starting_vocab=vocab, add_if_not_present=True, verbose=verbose
    )

    logger.info(f'Created and tokenized dataset with {len(response_tokens)} examples')

    if test_split > 0:
        n_examples = len(context_tokens)
        n_test = int(test_split * n_examples)
        choice = np.random.choice(range(n_examples), size=n_test, replace=False)
        test_idx = np.zeros(n_examples, dtype=bool)
        test_idx[choice] = True
        train_idx = ~test_idx

        train_context = context_tokens[train_idx]
        train_response = response_tokens[train_idx]
        train_dir = os.path.join(output_filepath, 'train')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        np.savez(os.path.join(train_dir, 'msg_tokens'), X=train_context, Y=train_response)

        test_context = context_tokens[test_idx]
        test_response = response_tokens[test_idx]
        test_dir = os.path.join(output_filepath, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        np.savez(os.path.join(test_dir, 'msg_tokens'), X=test_context, Y=test_response)

    else:
        np.savez(os.path.join(output_filepath, 'msg_tokens'), X=context_tokens, Y=response_tokens)

    # Save vocab, pp means post processing
    vocab.save_vocab(os.path.join(output_filepath, 'vocab_pp.json'))

    return


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
