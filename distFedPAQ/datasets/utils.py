from beartype import beartype

import numpy as np
from numpy.typing import NDArray

__all__ = ["data_splitter", "generate_dummy_data", "pack_data"]


@beartype
def data_splitter(data: NDArray, n: int):
    """
    Split uniformly at random the data into `n` (almost) equitable datasets.

    Parameters
    ----------
    data : NDArray
        the data to split
    n : int
        the number of splits

    Returns
    -------
    List|NDArray
        the data (for `n=1`) or a list of `n` smaller data
    """
    full_size = data.shape[0]
    assert n > 0, f"number of splits must be a positive number, {n} were given"
    assert (
        n <= full_size
    ), f"number of splits must be less than the size of the data, {n}>{full_size} were given"
    if n == 1:
        return data

    output = []
    selector = np.arange(full_size, dtype=int)
    np.random.shuffle(selector)

    batch_size = full_size // n
    extra = full_size % n

    for i in range(n):
        start = i * batch_size
        end = (i + 1) * batch_size + extra
        extra = 0  # no extra for the rest
        output.append(data[selector[start:end]])

    return output


@beartype
def generate_dummy_data(size: int, dim: int = 2):
    """
    Dummy `X,y` data (regression) generator.

    Parameters
    ----------
    size : int
        number of sample
    dim : int, optional
        dimension of `X`, by default 2

    Returns
    -------
    Tuple[NDArray, NDArray]
        `X, y`: the data and its class
    """
    mu = np.random.randint(-10, 10)
    sigma = 10 * np.random.rand()
    X = mu + sigma * np.random.randn(size, dim)
    W = np.random.randint(-2, 2, size=(dim, 1)) + np.random.randn(dim, 1)
    noise = np.random.randn(size, 1)
    y = X @ W + noise
    return X, y


@beartype
def pack_data(X: NDArray, y: NDArray, add_ones: bool = True):
    """
    Concatenate the input and target data into one big array.

    Parameters
    ----------
    X : NDArray
        `n x d1` input data, `n` samples of `d1` features
    y : NDArray
        `n x d2` target data, `n` samples of `d2` features
    add_ones : bool, optional
        a boolean flag on either a column of ones will be added between `X` and `y` or not, by default True

    Returns
    -------
    Tuple[NDArray, Tuple[int,Lite]]
        _description_
    """
    if len(y.shape) == 1:
        y = y.copy().reshape(-1, 1)

    n, d1 = X.shape
    d2 = y.shape[1]

    if add_ones:
        full_data = np.hstack([X, np.ones((n, 1)), y])
        weight_size = (d1 + 1, d2)
    else:
        full_data = np.hstack([X, y])
        weight_size = (d1, d2)

    return full_data, weight_size
