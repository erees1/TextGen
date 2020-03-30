import argparse
from src.cli import chat
from src.cli import respond
import logging
from dotenv import find_dotenv, load_dotenv

def create_argument_parser():
    parser = argparse.ArgumentParser(
        prog="TextGen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Command line interface to interact with trained chatbot "
    )

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parsers = [parent_parser]

    subparsers = parser.add_subparsers(help="Available TextGen commands:")

    chat.add_subparser(subparsers, parents=parent_parsers)
    respond.add_subparser(subparsers, parents=parent_parsers)

    return parser


def main():
    log_fmt = '%(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    # Parse command line arguments
    arg_parser = create_argument_parser()
    cmdline_arguments = arg_parser.parse_args()

    if hasattr(cmdline_arguments, 'func'):
        cmdline_arguments.func(cmdline_arguments)
    else:
        arg_parser.print_help()
        exit(1)
