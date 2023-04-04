from beartype.typing import Tuple, List, Callable

from argparse import ArgumentParser
import numpy as np
from numpy.typing import NDArray

from distFedPAQ.functional.tools import parallel_trainer, sequential_trainer

__all__ = ["arg_parser"]

__description = """
Test the distFedPAQ module. 
In order to play with it, user needs to run python in interactive mode by running the command
$ python3 -i main.py -n ...
"""


def arg_parser() -> Tuple[
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    bool,
    Callable[..., List[float]],
    NDArray,
    int,
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
        default=10,
        help="number of iterations per node for each local update. Default to 10.",
    )
    parser.add_argument(
        "--external-update",
        "-ext",
        type=int,
        default=100,
        help="number of time that some nodes (see -n-ext-ave) will update externally by averaging. Default to 100.",
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
        help="size of a batch from local data of a node in each iteration of its local update. Default to 1.",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-3,
        help="learning rate for nodes' local update. Default to 1e-3.",
    )
    parser.add_argument(
        "--momentum",
        "-mom",
        type=float,
        default=0,
        help="momentum for a local update. Default to 0.",
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
        "--repeat-single",
        "-rs",
        type=int,
        default=1,
        help="number of time that the 'single node' will do loops of local updates before evaluation. 1",
    )
    parser.add_argument(
        "--path-to-probability-P",
        "-P-path",
        type=str,
        help="path to the probability file which encode the graph. If not set, probability will be uniform on all nodes.",
    )

    # > argument fetcher from parser
    args = vars(parser.parse_args())

    n = args["nodes"]
    n_loc_update = args["local_update"]
    n_ext_update = args["external_update"]
    nodes_external_averaging = args["nodes_external_averaging"]
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    mom = args["momentum"]
    add_ones = args["with_bias"] == "y"
    tr = args["trainer"]
    rs = args["repeat_single"]
    path_proba = args["path_to_probability_P"]

    # > checkers
    assert n > 0, f"n ({n}) must be a positive integer"
    assert n_loc_update > 0, f"n_loc_update ({n_loc_update}) must be a positive integer"
    assert batch_size > 0, f"batch size ({batch_size}) must be a positive integer"
    assert lr > 0, f"learning rate ({lr}) must be a positive float"
    assert 0 <= mom and mom < 1, f"momentum ({mom}) must be in [0,1)"
    assert (
        0 < nodes_external_averaging and nodes_external_averaging <= n
    ), f"n-ext-ave ({nodes_external_averaging}) must be positive and cannot exceed n ({n})"
    assert tr in [
        "seq",
        "sequential",
        "par",
        "parallel",
    ], f"trainer ({tr}) must be one of 'seq, sequential, par, parallel'."
    assert (
        rs > 0
    ), f"number of time  ({rs}) that the single node will be repeated must be a positive integer"
    # > end checkers

    trainer = {True: parallel_trainer, False: sequential_trainer}.get("par" in tr)

    if path_proba is not None:
        P = np.genfromtxt(path_proba, dtype=float, delimiter=",", filling_values=0)
        assert (
            len(P.shape) == 2
        ), f"Probability must be a 2 dim array, {len(P.shape)} dim were given."
        assert (
            P.shape[0] == P.shape[1] and P.shape[0] == n
        ), f"Probability must be a {n}x{n} square matrix, {P.shape[0]}x{P.shape[1]} were given."

        print(P)
    else:
        P = 1 / (n - 1) * (np.ones((n, n)) - np.eye(n))
        # For n = 5
        # P = array([[0.  , 0.25, 0.25, 0.25, 0.25],
        #    [0.25, 0.  , 0.25, 0.25, 0.25],
        #    [0.25, 0.25, 0.  , 0.25, 0.25],
        #    [0.25, 0.25, 0.25, 0.  , 0.25],
        #    [0.25, 0.25, 0.25, 0.25, 0.  ]])

    return (
        n,
        n_loc_update,
        n_ext_update,
        nodes_external_averaging,
        batch_size,
        lr,
        mom,
        add_ones,
        trainer,
        P,
        rs,
    )
