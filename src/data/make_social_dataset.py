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
from spellchecker import SpellChecker
from tqdm import tqdm
import math
import yaml
from sklearn.model_selection import train_test_split
from itertools import compress
from tensorflow import keras

# Global variables
verbose = 1


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('spec_filepath', type=click.Path())
@click.option('--test_split', default=0.0, type=float)
def main(input_filepath, output_filepath, spec_filepath, test_split):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # parse specs from spec_filepath
    with open(spec_filepath) as f:
        specs = yaml.load(f, Loader=yaml.FullLoader)

    socialml_dataset = extract_socialml_dataset(input_filepath, output_filepath, specs)
    context_strings = socialml_dataset[0]
    response_strings = socialml_dataset[1]

    data = [
        [context, response]
        for context, response in zip(context_strings, response_strings)
        if context != '' and response != '']

    context_strings = [i[0] for i in data]
    response_strings = [i[1] for i in data]

    # Add tags
    start_tag = specs['tags'][specs['tokens']['START_TOKEN']]
    end_tag = specs['tags'][specs['tokens']['END_TOKEN']]
    response_strings = list(map(lambda x: add_tags(x, start_tag, end_tag), response_strings))

    # Remove unwanted characters
    unwanted_chars = ['\n', '\t']

    def _remove_chars(i):
        return remove_chars(i, unwanted_chars)

    context_strings = list(map(_remove_chars, context_strings))
    response_strings = list(map(_remove_chars, response_strings))

    strings = [
        line1 + '--$--' + line2 + '\n'
        for line1, line2 in zip(context_strings, response_strings)]

    if test_split > 0:
        train, test = train_test_split(
            strings,
            test_size=test_split,
        )

        train_dir = os.path.join(output_filepath, 'train')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        save_txt(train, output_filepath=os.path.join(train_dir, 'social.txt'))

        test_dir = os.path.join(output_filepath, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        save_txt(test, output_filepath=os.path.join(test_dir, 'social.txt'))
    else:
        save_txt(strings, output_filepath=os.path.join(output_filepath, 'social.txt'))

    # Create and save vocab from response_strings
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(response_strings)
    datastore = tokenizer.to_json()
    with open(os.path.join(output_filepath, 'vocab.json'), 'w') as f:
        json.dump(datastore, f)


def extract_socialml_dataset(input_filepath, output_filepath, specs):
    '''Extract messages using the socialml package
    '''

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

    # Convert the array to dataset pairs of (context, reponse)
    message_dataset = socialml.make_training_examples(
        message_array,
        max_context_length=specs['max_context'],
        combine_contexts=specs['combine_contexts'],
        add_seq_tags=False,
        verbose=1,
    )

    return message_dataset


def save_data(X, Y, output_filepath):
    if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        np.savez(output_filepath, X=X, Y=Y)
    elif isinstance(X, list) and isinstance(Y, list):
        joblib.dump(X, os.path.join(output_filepath + '_X.gz'))
        joblib.dump(X, os.path.join(output_filepath + '_Y.gz'))


def save_txt(X, output_filepath='./'):
    with open(output_filepath, 'w') as f:
        f.writelines(X)

def add_tags(x, start_tag, end_tag):
    return ' '.join([start_tag, x, end_tag])

def remove_chars(string, char_list):
    x = {i: ' ' for i in char_list}
    transtab = str.maketrans(x)
    string = string.translate(transtab).strip()
    while '  ' in string:
        string = string.replace('  ', ' ')
    return string

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
