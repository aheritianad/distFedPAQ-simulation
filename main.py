from distFedPAQ.nodes import Node
from distFedPAQ.datasets import pack_data, data_splitter, generate_dummy_data
from distFedPAQ.utils.plots import custom_plot
from distFedPAQ.utils.argument_parser import arg_parser
from distFedPAQ.functional.loss_functions import mse_loss, grad_mse_loss
from distFedPAQ.functional.tools import repeat

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


from tqdm import tqdm


if __name__ == "__main__":
    # > PARAMS FETCHING
    (
        n,
        n_loc_update,
        n_ext_update,
        n_nodes_ext_ave,
        batch_size,
        lr,
        momentum,
        add_ones,
        trainer,
        single_times,
    ) = arg_parser()

    # > DATA LOADING AND PREPARATION
    X, y = load_diabetes(return_X_y=True)
    # X, y = generate_dummy_data(size=300)
    full_data, weight_size = pack_data(X, y, add_ones)
    train_data, test_data = train_test_split(full_data, test_size=0.2)

    # > NODES DEFINITION
    # #> single
    single_node = Node(
        data=data_splitter(train_data, 1),
        weight_size=weight_size,
        grad_f=grad_mse_loss,
        batch_size=batch_size,
    )
    # #> multiple
    nodes = np.array(
        [
            Node(
                data=data,
                weight_size=weight_size,
                grad_f=grad_mse_loss,
                batch_size=batch_size,
            )
            for data in data_splitter(train_data, n)
        ],
        dtype=Node,
    )
    # > OUTPUTS PREPARATIONS
    # ~ test_sample = lambda: np.vstack([node.local_data.sample() for node in nodes])
    losses_train = np.empty((n + 1, n_ext_update + 1))
    losses_test = np.empty((n + 1, n_ext_update + 1))
    print("\n\n\n")
    # > TRAINING
    # #> Single node training
    print("--- SINGLE NODE TRAINING ---\n")

    single_node.local_update(n_loc_update, lr)
    losses_train[0, 0] = mse_loss(train_data, single_node.w)
    losses_test[0, 0] = mse_loss(test_data, single_node.w)
    for j in tqdm(range(n_ext_update)):
        for i in range(single_times):
            single_node.local_update(n_loc_update)
        # test_data = test_sample()
        losses_train[0, j + 1] = mse_loss(train_data, single_node.w)
        losses_test[0, j + 1] = mse_loss(test_data, single_node.w)

    # #> Multiple nodes training

    print(f"\n--- MULTIPLE ({n}) NODES TRAINING ---\n")

    # ~test_data = test_sample()
    # ##> first local update
    outputs = trainer(
        nodes=nodes,
        n_iters=repeat(n_loc_update, n),
        lrs=repeat(lr, n),
        eval_data=repeat([train_data, test_data], n),
        loss_fn=mse_loss,
    )
    for i, (loss_train, loss_test) in enumerate(outputs):
        losses_train[i + 1, 0] = loss_train
        losses_test[i + 1, 0] = loss_test

    ave = dict()
    for j in tqdm(range(n_ext_update)):
        # ##> external averaging
        selector = np.random.choice(n, size=n_nodes_ext_ave, replace=False)
        for s in selector:
            ave[s] = ave.get(s, 0) + 1
        Node.external_update(*nodes[selector])

        # ###b> local update
        # ~ test_data = test_sample()
        outputs = trainer(
            nodes=nodes,
            n_iters=repeat(n_loc_update, n),
            lrs=repeat(lr, n),
            eval_data=repeat([train_data, test_data], n),
            loss_fn=mse_loss,
        )
        for i, (loss_train, loss_test) in enumerate(outputs):
            losses_train[i + 1, j + 1] = loss_train
            losses_test[i + 1, j + 1] = loss_test
    print(f"number of time a node made an external update :")
    for i, k in sorted(ave.items(), key=lambda x: (x[1], -x[0]), reverse=True):
        print(f"\tNode_{i+1} -> {k} times.")

    # > INSTRUCTIONS
    print(
        f""""
        
        --- INSTRUCTION FOR PLOT FUNCTION ---
        
        >>> plot(1,3,5) # plot nodes 1,3,5 evaluations. It can take as many value as nodes available
                        # note that 0 stands for single node.     
                        
        >>> plot(start=2,stop=5) # plot evaluations for node 2,3,4,5 i.e. from start to stop (inclusive)
        >>> plot(start= 3)# will not do anything unless stop is specified. start default value is 1.
        >>> plot(stop=5) # plot evaluations for node 1,2,3,4,5 i.e. from 1 to stop (inclusive)
        >>> plot(stop=-1) # plot evaluations for each node start to node LAST i.e. ALL nodes
        >>> plot(start=2, stop=2) # plot evaluations for SECOND node to SECOND LAST node. 
        
        >>> plot(ave = False) # hide the plot of average of the nodes
        >>> plot(single = False) # hide the plot of single node
        >>> plot(train = False) # hide evaluation with train data
        >>> plot(test = False) # hide evaluation with test data
        
        USER CAN USE ANY COMBINATION OF THESE BUT FOLLOWING THIS ORDER IS RECOMMENDED.
        eg : 
            >>> plot(1,2,3, ave=False, single=False, train=False, test=True) # show eval of node 1, node2 and node3 on test data
            >>> plot(start=1,stop=3, ave=False, single=False, train=False) # the same as above
        NB : single node got trained {single_times} x more than others node before evaluations.

        """
    )
    # > SETUPS FOR PLOTTING

    title = f"""
    Value of $f$ for a local update per iteration
    nb of nodes = {n}
    batch size = {batch_size}
    nb iter for loc update = {n_loc_update}
    learning rate {lr}
    """

    def plot(
        *indices, start=1, stop=None, single=True, ave=True, train=True, test=True
    ):
        if train:
            custom_plot(
                losses_train,
                *indices,
                i_start=start,
                i_stop=stop,
                single=single,
                ave=ave,
                pref="train ",
                style1="-",
                style2="--",
            )
        if test:
            custom_plot(
                losses_test,
                *indices,
                i_start=start,
                i_stop=stop,
                single=single,
                ave=ave,
                pref="test ",
                style1="-.",
                style2=":",
            )
        # ##> LEGEND
        plt.xlabel("iteration")
        plt.legend()
        plt.title(title)
        plt.show()

    plot(stop=-1)
