from beartype import beartype
from beartype.typing import Optional


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
        np.random.shuffle(self.__selector)
        self.__sampler_index = 0

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
        np.random.shuffle(self.__selector)
        self.__sampler_index = 0

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
    def __getitem__(self, size: int):
        """
        Retrieving `size` sample;

        Parameters
        ----------
        size : int
            sample size

        Returns
        -------
        NDArray
            a sample of size `size`
        """
        return self.sample()

    def sample(self, size: Optional[int] = None):
        """
        Retrieving a sample; same as `sample` method.

        Returns
        -------
        NDArray
            a sample
        """
        if size is None:
            size = self.__batch_size

        start = self.__sampler_index
        self.__sampler_index += size
        selected = self.__selector[start : self.__sampler_index]
        if self.size[0] <= self.__sampler_index:
            self.__sampler_index = 0
            np.random.shuffle(self.__selector)
        return self.__data[selected]
