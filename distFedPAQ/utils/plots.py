import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, Iterable

__all__ = ["custom_plot"]


@beartype
def custom_plot(
    losses: Iterable,
    *indices,  # g: Iterable[int],
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
            plt.plot(losses[i], style2, label="{1}$Node_{0}$".format(i, pref))
        except IndexError:
            break


def graph(P, ave=None):
    G = nx.from_numpy_array(P)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, labels={i: i for i in G.nodes})
    e_label = {
        (i, j): f"proba {(P[i][j],P[j][i])}" for i, j in zip(*np.nonzero(P)) if i <= j
    }
    if ave is not None:
        for i, j in G.edges:
            e_label[(i, j)] += f" contact: {ave[i][j] + ave[j][i]}"

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=e_label,
    )
