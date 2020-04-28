# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import shutil
import argparse
import yaml
from src.models import tf_dataset_loader, seq2seq
from src.data.word_utils import Vocab


def build_and_train(
    X_data_path,
    Y_data_path,
    spec_path='',
    log_dir='',
    checkpoint_dir='',
    vocab_filepath='',
    word2vec_filepath='',
):

    # Load specs from yaml file
    specs = load_spec(spec_path)
    print('spec:', specs)
    tf_dataset_params = specs['tf_dataset_params']
    model_params = specs['model_params']
    training_params = specs['training_params']

    # Conver npz array into tf.dataset and batch and shuffle etc
    dataset = tf_dataset_loader.load_from_txt(
        X_data_path, Y_data_path, word2vec_filepath, vocab_filepath, **tf_dataset_params
    )

    # Build and compile the model
    vocab = Vocab(vocab_filepath)
    model, _, _ = seq2seq.build_model(**model_params, vocab_size=len(vocab))
    model = seq2seq.compile_model(model)

    callbacks = get_callbacks(log_dir, checkpoint_dir)
    # Save vocab as of training so we have a copy incase dataset changes
    vocab.save_vocab(os.path.join(checkpoint_dir, 'vocab.json'))

    # Train the model
    fit_model(model, dataset, callbacks, **training_params)

    return


def fit_model(model, dataset, callbacks, epochs):
    model.fit(dataset, epochs=epochs, callbacks=callbacks)
    return model


def get_callbacks(logdir, checkpoint_dir):
    clean_and_create_dir(checkpoint_dir)
    clean_and_create_dir(logdir)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    # Checkpoint callbacks
    checkpoint = os.path.join(checkpoint_dir, "model-weights-epoch-{epoch:02d}-loss-{loss:.4f}.hdf5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint,
        save_weights_only=True,
        monitor='loss',
        save_best_only=True,
        mode='min',
    )

    return [tensorboard_callback, checkpoint_callback]


def clean_and_create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)


def load_spec(spec_path):

    if spec_path is not None:
        pass
    else:
        print('Must provide spec path')
        exit(1)

    with open(spec_path) as f:
        specs = yaml.load(f, Loader=yaml.Loader)

    parsed_specs = {}
    for param_type in specs.keys():
        params = specs[param_type]
        parsed_params = {}
        for key, value in params.items():
            try:
                parsed_params[key] = int(value)
            except Exception:
                parsed_params[key] = value
        parsed_specs[param_type] = parsed_params

    return parsed_specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('X_data_path', type=str, default=None)
    parser.add_argument('Y_data_path', type=str, default=None)
    parser.add_argument('--spec_path', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--vocab_filepath', type=str, default=None)
    parser.add_argument('--word2vec_filepath', type=str, default=None)

    args = parser.parse_args()
    arg_vars = vars(args)

    build_and_train(**arg_vars)

    exit(0)


if __name__ == '__main__':
    main()
