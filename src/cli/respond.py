# from src.models import predict_model
import argparse
from typing import List, Text
from src.cli import utils
from src.models import predict_model


def add_subparser(subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]):
    respond_parser = subparsers.add_parser(
        "respond",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Respond to single message",
    )

    respond_parser.set_defaults(func=respond)
    respond_parser.add_argument('Message', help='Message for the model to repsond to')
    respond_parser = utils.create_model_loader_args(respond_parser)
    respond_parser = utils.create_inference_args(respond_parser)


def respond(args):
    model, encoder_model, decoder_model, vocab_filepath, model_spec_filepath = utils.load_model_from_args(args)
    data_spec_filepath = args.data_specs
    seq_inf = predict_model.SeqInference(
        vocab_filepath,
        encoder_model,
        decoder_model,
        model_spec_filepath,
        data_spec_filepath,
        verbose=0,
        method=args.method,
    )
    out = seq_inf.predict_response_from_text(args.Message)
    print(out)
