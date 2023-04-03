from distFedPAQ.nodes import Node
from numpy.typing import NDArray

from beartype import beartype
from beartype.typing import Optional, Any, Callable, Iterable, Literal, List
import multiprocess as mp


__all__ = [
    "repeat",
    "run_in_parallel",
    "run_in_parallel_multi_args",
    "parallel_trainer",
    "sequential_trainer",
]


@beartype
def repeat(value: Any, n: int):
    assert n > 0, "n must be a positive natural number"
    return [value] * n


@beartype
def run_in_parallel(function: Callable, args: Iterable, p: Optional[int] = None):
    with mp.Pool(processes=p) as pool:
        result = pool.map(function, args)
    return result


@beartype
def run_in_parallel_multi_args(
    function: Callable, args: Iterable[Iterable], p: Optional[int] = None
):
    with mp.Pool(processes=p) as pool:
        result = pool.starmap(function, args)
    return result


@beartype
def _local_updater(node: Node, n_iter: int, lr: float) -> NDArray:
    node.local_update(n_iter, lr)
    return node.w


@beartype
def parallel_trainer(
    nodes: Iterable[Node],
    n_iters: Iterable[int],
    lrs: Iterable[float],
    eval_data: Optional[List[List[NDArray]]] = None,
    loss_fn: Optional[Callable[..., float]] = None,
    p: Optional[int] = None,
):

    # #> new weights computation
    new_weights = run_in_parallel_multi_args(
        function=_local_updater, args=zip(nodes, n_iters, lrs), p=p
    )
    # #> weights update
    for node, new_w in zip(nodes, new_weights):
        node.w = new_w

    # #> losses computation
    if eval_data is not None and loss_fn is not None:

        losses = [
            run_in_parallel_multi_args(
                function=loss_fn,
                args=zip(data, new_weights),
                p=p,
            )
            for data in eval_data
        ]
        return losses


@beartype
def sequential_trainer(
    nodes: Iterable[Node],
    n_iters: Iterable[int],
    lrs: Iterable[float],
    eval_data: Optional[List[List[NDArray]]] = None,
    loss_fn: Optional[Callable[..., float]] = None,
    p: Literal[None] = None,  # place holder for processors
):
    # #> weights update
    for node, n_iter, lr in zip(nodes, n_iters, lrs):
        node.local_update(n_iter, lr)

    # #> losses computation
    if eval_data is not None and loss_fn is not None:
        losses = [
            [loss_fn(d, node.w) for d in data] for node, data in zip(nodes, eval_data)
        ]
        return losses
