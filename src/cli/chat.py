# from src.models import predict_model
import argparse
from typing import List, Text
from src.cli import utils
from src.models import predict_model
import sys
from termcolor import cprint


def add_subparser(subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]):
    chat_parser = subparsers.add_parser(
        "chat",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Chat to trained chatbot",
    )

    chat_parser.set_defaults(func=chat)
    chat_parser = utils.create_model_loader_args(chat_parser)
    chat_parser = utils.create_inference_args(chat_parser)
    chat_parser.add_argument(
        '--max_context', default=7, help='Number of messages to provide in the context for a response'
    )


def chat(args):
    model, encoder_model, decoder_model, vocab_filepath, model_spec_filepath = utils.load_model_from_args(args)
    data_spec_filepath = args.data_specs
    chat = predict_model.Chat(
        vocab_filepath,
        encoder_model,
        decoder_model,
        model_spec_filepath,
        data_spec_filepath,
        verbose=0,
        method=args.method,
        max_context=args.max_context,
    )
    print()
    cprint('Bot Loaded. Type a message and press enter (use quit() to exit)', 'green')
    while True:
        msg = get_input('Your input --> ')
        print(msg)

        if str(msg) == 'quit()':
            print('Goodbye!')
            exit(0)
            return
            break
        else:
            response = chat.send(msg)
            cprint(f'{response}', 'cyan')


def get_input(prompt):
    msg = input(prompt)
    spaces = ' ' * (len(msg) + len(prompt) + 1)
    print(f"\033[A{spaces}\033[A")
    return msg
