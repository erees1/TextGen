# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import shutil
import argparse
import yaml
from src.data_transform import preprocessing as pp
from src.models import lstm_seq
from src.utils.specs import load_spec
import json
import logging


def build_and_train(
    model_name,
    data_path,
    spec_path='',
    log_dir='',
    checkpoint_dir='',
    **kwargs,
):
    model_names = ['lstm_seq']
    if model_name not in model_names:
        raise (NotImplementedError('Model name not in avaiable list'))
    if model_name == 'lstm_seq':
        Model = lstm_seq.LSTMSeqModel

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Load specs from yaml file
    specs = load_spec(spec_path)
    logger.info(f'spec: \n {json.dumps(specs)}')
    tf_dataset_params = specs['tf_dataset_params']
    model_params = specs['model_params']
    training_params = specs['training_params']

    # Load data from text
    dataset = pp.load_from_txt(data_path)

    # Preprocess the data
    logger.info(f'Preprocessing dataset')
    dataset, integer_encoder = pp.preprocess_data(dataset, **specs['tf_dataset_params'])

    # Build and compile the model
    vocab_list = integer_encoder.tokens
    logger.info(f'Instantiating model {Model}')
    model = Model(vocab_list, **model_params)

    callbacks = _get_callbacks(log_dir, checkpoint_dir)

    # Save vocab as of training so we have a copy incase dataset changes
    with open(os.path.join(checkpoint_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab_list, f)

    # Train the model
    _fit_model(model, dataset, callbacks, **training_params, **kwargs)

    return


def _fit_model(model, dataset, callbacks, epochs, loss, **kwargs):

    learning_rate = CustomSchedule(256)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=optimizer, loss=loss)


    model.fit(dataset, epochs=epochs, callbacks=callbacks, **kwargs)
    return model


def _get_callbacks(logdir, checkpoint_dir):
    _clean_and_create_dir(checkpoint_dir)
    _clean_and_create_dir(logdir)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)

    # Checkpoint callbacks
    checkpoint = os.path.join(checkpoint_dir, "model-weights-epoch-{epoch:02d}-loss-{loss:.4f}.tf")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint,
        save_weights_only=True,
        monitor='loss',
        save_best_only=True,
        mode='min',
    )

    return [tensorboard_callback, checkpoint_callback]


def _clean_and_create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, default=None)
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
