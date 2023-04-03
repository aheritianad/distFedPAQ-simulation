from beartype import beartype
from beartype.typing import Any

# from math import ceil

import numpy as np
from numpy.typing import NDArray

__all__ = ["Data"]


class Data:
    @beartype
    def __init__(self, content: NDArray, batch_size: int):
        self.__data = content.copy()
        self.__size = content.shape
        self.__batch_size = batch_size
        self.__selector = np.arange(self.size[0])
        # self.__n_batches = ceil(self.size[0] / self.__batch_size)
        # self.__samples = self.__sampler()

    @property
    def size(self):
        """
        Size/shape of the data.

        Returns
        -------
        Tuple[int]|int
            dimension of the data
        """
        return self.__size

    @property
    def batch_size(self):
        """
        size of a sample.

        A sample might be less than this when there are not enough data at the time of sampling.

        Returns
        -------
        int
            size of a batch
        """
        return self.__batch_size

    @beartype
    @batch_size.setter
    def batch_size(self, value: int):
        """`batch_size` updater"""

        assert value > 0, "Batch size must be a positive integer"
        self.__batch_size = value
        # self.__n_batches = ceil(self.size[0] / self.__batch_size)
        # self.__samples = self.__sampler()

    @property
    def content(self):
        """
        Retrieve a copy of the content.

        Returns
        -------
        NDArray
            a copy of the real data
        """
        return self.__data.copy()

    @beartype
    @content.setter
    def content(self, value: NDArray):
        """
        Content updater

        Parameters
        ----------
        value : NDArray
            a new data value
        """
        self.__data = value.copy()
        self.__size = value.shape[0]
        self.__selector = np.arange(self.size[0])
        # self.__n_batches = ceil(self.size[0] / self.__batch_size)
        # self.__samples = self.__sampler()

    def __len__(self):
        """
        Number of sample/row in the data

        Returns
        -------
        int
            length of the data
        """
        return self.size[0]

    @beartype
    def __getitem__(self, index: Any):
        """
        Retrieving a sample; same as `sample` method.

        Parameters
        ----------
        index : Any
            will not be considered

        Returns
        -------
        NDArray
            a sample
        """
        return self.sample()

    def sample(self):
        """
        Retrieving a sample; same as `sample` method.

        Returns
        -------
        NDArray
            a sample
        """
        np.random.shuffle(self.__selector)
        start = 0  # i * self.__batch_size
        stop = self.__batch_size  # (i + 1) * self.__batch_size
        idx = self.__selector[start:stop]
        return self.__data[idx]
        # return next(self.__samples)

    # def __sampler(self):
    #     """
    #     Sample generator for the data

    #     Yields
    #     ------
    #     NDArray
    #         a sample
    #     """
    #     while True:
    #         np.random.shuffle(self.__selector)
    #         for i in range(self.__n_batches):
    #             start = i * self.__batch_size
    #             stop = (i + 1) * self.__batch_size
    #             idx = self.__selector[start:stop]
    #             yield self.__data[idx]
