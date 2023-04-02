import numpy as np
from numpy.typing import NDArray

from beartype import beartype


@beartype
def mse_loss(Xy: NDArray, w: NDArray) -> float:
    d = w.shape[0]
    x = Xy[:, :d]
    y = Xy[:, d:]
    return np.mean(np.linalg.norm(x @ w - y, axis=1) ** 2)


@beartype
def grad_mse_loss(Xy: NDArray, w: NDArray) -> NDArray:
    d = w.shape[0]
    x = Xy[:, :d]
    y = Xy[:, d:]
    N = x.shape[0]
    return 2 * x.T @ (x @ w - y) / N
