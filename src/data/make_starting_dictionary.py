# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from spellchecker import SpellChecker
import json
import os
import yaml


@click.command()
@click.argument('save_dir', type=click.Path())
@click.argument('spec_filepath', type=click.Path())
def main(save_dir, spec_filepath):

    # Parse specification file
    with open(spec_filepath) as f:
        specs = yaml.load(f, Loader=yaml.BaseLoader)
    TAGS = specs['tags']
    DICTIONARY_LENGTH = int(specs['dictionary_length'])

    # Dictionary is from spell checker module
    sc = SpellChecker()
    dictionary = sc.word_frequency.dictionary
    short_dictionary = dict(dictionary.most_common(DICTIONARY_LENGTH))

    # save dictionary file words along with word counts
    with open(os.path.join(save_dir, '40k_dictionary.json'), 'w') as f:
        json.dump(short_dictionary, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
