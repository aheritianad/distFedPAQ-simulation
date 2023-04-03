from __future__ import annotations

# ~import time
from distFedPAQ.datasets import Data

import numpy as np
from numpy.typing import NDArray

from beartype import beartype
from beartype.typing import Union, Tuple, List, Optional, Callable

__all__ = ["Node"]


class Node:
    @beartype
    def __init__(
        self,
        data: NDArray,
        weight_size: Union[Tuple, int],
        grad_f: Callable,
        batch_size: int = 1,
    ) -> None:
        """
        Trainer node with local `sgd` updater.

        Parameters
        ----------
        data : NDArray
            local dataset for the node
        weight_size : Union[Tuple, int]
            size of the weight of the objective function `f`.
        grad_f : Callable
            gradient of the objective function `f`. (will be use for sgd local update)
        batch_size : int, optional
            size of the batch of local dataset's sampler, by default 1
        lr : float, optional
            learning rate for sgd local update, by default 1e-2
        use_tqdm : bool, optional
            boolean flag for viewing local update, by default False
        """
        self.local_data = Data(content=data, batch_size=batch_size)

        self.__batch_size = batch_size

        self.grad_f = grad_f  # gradient of the objective function
        self.__w = 5 * np.random.randn(
            *weight_size
        )  # local weight for the objective function

        self.external_update = self.__external_update

    @property
    def data_batch_size(self):
        """
        Retrieve the value of `batch_size` of the sampler of the local dataset.

        Returns
        -------
        int
            the value of `batch_size` of the local dataset
        """
        return self.__batch_size

    @beartype
    @data_batch_size.setter
    def data_batch_size(self, value: int):
        """
        Update the `batch_size` for the sampler of the local dataset.

        Parameters
        ----------
        value : int
            new value for `batch_size`
        """
        self.__batch_size = value
        self.local_data.batch_size = value

    @property
    def w(self):
        return self.__w

    @beartype
    @w.setter
    def w(self, new_w: NDArray[np.float64]):
        assert (
            new_w.shape == self.__w.shape
        ), f"Incorrect shape for new weight. {self.__w.shape} expected but {new_w.shape} were given."
        self.__w = new_w

    @beartype
    def local_update(self, n_iter: int, lr: float = 1e-4):
        """
        Update the local parameter `w` with `sgd` on the local dataset.

        Parameters
        ----------
        n_iter : int
            Number of iterations for the update.
        """
        for _ in range(n_iter):
            x = self.local_data.sample()
            self.__w -= lr * self.grad_f(x, self.__w)

        # time.sleep(0.001)

    @beartype
    def __external_update(
        self,
        *others,
        weight: Optional[float] = None,
        others_weights: Optional[List[float]] = None,
    ):
        """
        Update the parameter for the node and other nodes by (weighted) averaging.

        It is a copy of the `static method` called `external_update`.
        Parameters
        ----------
        weight : Optional[float], optional
            the weight for the current node, by default None
        others_weights : Optional[List[float]], optional
            weight of all the other nodes, by default None
        """
        if weight is not None and others_weights is not None:
            ave_weights = [weight, *others_weights]
        else:
            ave_weights = None

        Node.external_update(self, *others, ave_weights=ave_weights)

    @beartype
    @staticmethod
    def external_update(*nodes: Node, ave_weights: Optional[List[float]] = None):
        """
        Update the parameter `w` of each `node` of `nodes` with their (weighted) average.

        Parameters
        ----------
        ave_weights : Optional[List[float]], optional
            Weight to be used during averaging, by default None.
        """
        if ave_weights is not None:
            assert len(nodes) == len(
                ave_weights
            ), "Length of ave_weights must be equal to the number of nodes."
            assert all(
                [weight > 0 for weight in ave_weights]
            ), "Weights must be a positive number."
        else:
            ave_weights = [1] * len(nodes)

        total_weight = sum(ave_weights)

        w = np.zeros_like(nodes[0].w)
        for i, node in enumerate(nodes):
            w += ave_weights[i] * node.w

        w /= total_weight

        for node in nodes:
            node.w = w
