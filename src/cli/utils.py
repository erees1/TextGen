import argparse
import os
import glob
import re
import numpy as np
from src.models import predict_model, seq2seq
from src.data.word_utils import Vocab
import logging

def create_model_loader_args(parser):
    parser.add_argument(
        '--model_specs',
        help='Experiment number to load model and spec for',
        default='highest',
    )
    parser.add_argument(
        '--model',
        help='Path to saved  model weights, must be compatiable with experiment spec file',
        default='latest',
    )
    parser.add_argument(
        '--data_specs',
        help='Path to yaml file for data specification',
        default='specs/data_specs.yaml',
    )

    return parser

def load_model_from_args(args):

    if args.model_specs == 'highest':
        # Find all experiment spec files
        experiment_specs = glob.glob("specs/exp*.yaml")
        # Get spec file with highest number
        file_nums = []
        for i in experiment_specs:
            experiment_nums = re.search("(\d+).yaml$", i)       # NOQA
            file_nums.append(int(experiment_nums.group(1)))
        specs_filepath = experiment_specs[np.argmax(file_nums)]

    else:
        specs_filepath = args.model_specs

    specs = predict_model.load_yaml(specs_filepath)

    exp = re.search("(exp\d+)", specs_filepath).group(1)        # NOQA
    checkpoint_dir = f"models/checkpoints/{exp}"
    vocab_filepath = os.path.join(checkpoint_dir, 'vocab.json')
    vocab = Vocab(vocab_filepath)

    model, encoder_model, decoder_model = seq2seq.build_model(**specs['model_params'], vocab_size=len(vocab))

    log(f'Loaded model using {specs_filepath}')

    if args.model == 'latest':
        weights_checkpoint_filepath = predict_model.latest_checkpoint(checkpoint_dir)
        log(f'Loaded latest weights from {weights_checkpoint_filepath}')
    else:
        weights_checkpoint_filepath = args.model
        try:
            log(f'Loaded model weights from {weights_checkpoint_filepath}')
        except Exception:
            log(f"Couldn't find model weights at {weights_checkpoint_filepath}")

    model.load_weights(weights_checkpoint_filepath)

    return model, encoder_model, decoder_model, vocab_filepath, specs_filepath


def log(message):
    logger = logging.getLogger(__name__)
    logger.info(message)

def create_inference_args(parser):
    parser.add_argument('--method', help='Inference method', default='beam_search')
    return parser
