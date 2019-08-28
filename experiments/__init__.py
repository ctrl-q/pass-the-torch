"""
This file is used to process command line arguments provided by the user.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ast import literal_eval as le
import os


# Common args for all experiments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("datapath", help="Path to dataset folder")
parser.add_argument("-e", "--max_epoch", type=le, default=200, help="Max epoch")
parser.add_argument("-l", "--lr", type=le, default=0.0001, help="Learning rate")
parser.add_argument("-p", "--max_patience", type=le, help="max_patience for early stopping")
parser.add_argument("-b", "--batch_size", type=le, default=128, help="Minibatch size")
parser.add_argument("-t", "--trials", type=le, default=1, help="Number or tuning iterations. 1 == no tuning")


def parse_args(parser):
    """
    parse command line arguments

    Args:
        parser: an ArgumentParser object storing the user input or default command line arguments

    Returns:
         common_args: the 7 common arguments (datapath, train_subset, max_epoch, lr, max_patience, batch_size, trials)
         args: all the arguments provided by the user
    """
    args = vars(parser.parse_args())
    print("-" * 75)
    print("ARGS:", args)
    print("-" * 75)
    for arg, value in args.items():
        if arg not in ("datapath", "file_name", "trials"):
            if not isinstance(value, (list, tuple)):
                args[arg] = [value]  # create dummy param for skopt

    common_args = [
        "datapath",
        "max_epoch",
        "lr",
        "max_patience",
        "batch_size",
        "trials"
    ]

    common_args = [args.pop(arg) for arg in common_args]

    return common_args, args


def mark_best(results, folder):
    """
    Mark the best results within a collection of results and save them into a given folder

    Args:
        results: a collection of results
        folder: the name of the folder where to save the best result
    """
    best = max(results, key=lambda x: x[1])
    params, score = results[results.index(best)]

    file_name = "-".join([
        str(param if not isinstance(param, float) else round(param, 2))
        for param in params
    ])

    path = os.path.join(folder, file_name + ".tar")
    new_file_name = "best-" + file_name
    new_path = os.path.join(folder, new_file_name + ".tar")

    print("-" * 75)
    print("BEST PARAMS:", new_file_name, "; SCORE:", score)
    os.replace(path, new_path)
