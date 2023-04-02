import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, Iterable

__all__ = ["custom_plot"]


@beartype
def custom_plot(
    losses: Iterable,
    *indices: Iterable[int],
    i_start: int = 1,
    i_stop: Optional[int] = None,
    single: bool = True,
    ave: bool = True,
    pref: str = "",
    style1: str = "-",
    style2: str = ":",
):

    # ##>r REFERENCES
    if single:
        plt.plot(losses[0], style1, label=f"{pref}single Node")
    if ave:
        plt.plot(np.mean(losses[1:], axis=0), style1, label=f"{pref}average // Nodes")

    # ##> INDICES

    if not indices and i_stop is not None:
        if i_stop < 0:
            i_stop = i_stop % len(losses) + 1
        indices = range(i_start, i_stop)

    for i in indices:
        try:
            plt.plot(losses[i], style2, label=r"{1}$Node_{0}$".format(i, pref))
        except IndexError:
            break

    # ##> LEGEND
    plt.xlabel("iteration")
    plt.legend()
