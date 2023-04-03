from beartype.typing import Tuple, List, Callable
from argparse import ArgumentParser
from distFedPAQ.functional.tools import parallel_trainer, sequential_trainer

__all__ = ["arg_parser"]

__description = """
Test the distFedPAQ module. 
In order to play with it, user needs to run python in interactive mode by running the command
$ python3 -i main.py -n ...
"""


def arg_parser() -> Tuple[
    int, int, int, int, int, float, bool, Callable[..., List[float]]
]:

    parser = ArgumentParser(description=__description)

    # > add arguments to parser
    parser.add_argument(
        "--nodes",
        "-n",
        type=int,
        required=True,
        help="number of nodes for the training",
    )
    parser.add_argument(
        "--local-update",
        "-loc",
        type=int,
        default=100,
        help="number of node iteration for local update. Default to 100.",
    )
    parser.add_argument(
        "--external-update",
        "-ext",
        type=int,
        default=100,
        help="number of time that some nodes (see nodes_ext_ave) will update externally by averaging. Default to 100.",
    )
    parser.add_argument(
        "--nodes-external-averaging",
        "-n-ext-ave",
        type=int,
        default=2,
        help="number nodes will update externally by averaging. Default to 2.",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=1,
        help="size of a batch for a local update. Default to 1.",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-4,
        help="learning rate for a local update. Default to 1e-4.",
    )

    parser.add_argument(
        "--with-bias",
        "-bias",
        type=str,
        default="y",
        help="Flag for using 'bias' for the model [y/n]. Default to y.",
    )
    parser.add_argument(
        "--trainer",
        "-tr",
        type=str,
        default="seq",
        help="trainer for the nodes [[seq/sequential]/[par/parallel]]. Default to seq",
    )
    parser.add_argument(
        "--single-times",
        "-st",
        type=int,
        default=1,
        help="Default to 1",
    )

    # > argument fetcher from parser
    args = vars(parser.parse_args())

    n = args["nodes"]
    n_loc_update = args["local_update"]
    n_ext_update = args["external_update"]
    nodes_external_averaging = args["nodes_external_averaging"]
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    add_ones = args["with_bias"] == "y"
    tr = args["trainer"]
    st = args["single_times"]

    # > checkers
    assert n > 0, f"n ({n}) must be a positive integer"
    assert n_loc_update > 0, f"n_loc_update ({n_loc_update}) must be a positive integer"
    assert batch_size > 0, f"batch size ({batch_size}) must be a positive integer"
    assert lr > 0, f"learning rate ({lr}) must be a positive float"
    assert (
        0 < nodes_external_averaging and nodes_external_averaging <= n
    ), f"n-ext-ave ({nodes_external_averaging}) must be positive and cannot exceed n ({n})"
    assert tr in [
        "seq",
        "sequential",
        "par",
        "parallel",
    ], f"trainer ({tr}) must be one of 'seq, sequential, par, parallel'."
    assert st > 0, f"st ({st}) must be a positive integer"
    # > end checkers

    trainer = {True: parallel_trainer, False: sequential_trainer}.get("par" in tr)

    return (
        n,
        n_loc_update,
        n_ext_update,
        nodes_external_averaging,
        batch_size,
        lr,
        add_ones,
        trainer,
        st,
    )
