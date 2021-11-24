"""
To scale the run_deposit.py script, we had to forgo creating 
NX graph in memory. This script does exactly that. The motivation
is to isolate the high memory parts to a single file.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Any, List, Tuple


def main(args: Any):
    data: pd.DataFrame = pd.read_csv(args.data_file)

    print('making user graph...',  end = '', flush=True)
    graph: nx.DiGraph = make_graph(data.user, data.deposit)
    
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    plt.savefig(
        os.path.join(args.save_path, 'fig.png'), 
        bbox_inches="tight",
    )


def make_graph(node_a: pd.Series, node_b: pd.Series) -> nx.DiGraph:
    """
    DEPRECATED: This assumes we can store all connections in memory.

    Make a directed graph connecting each row of node_a to the 
    corresponding row of node_b.
    """
    assert node_a.size == node_b.size, "Dataframes are uneven sizes."

    graph = nx.DiGraph()

    nodes: np.array = np.concatenate([node_a.unique(), node_b.unique()])
    edges: List[Tuple[str, str]] = list(
        zip(node_a.to_numpy(), node_b.to_numpy())
    )

    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str, help='path to cached out of deposit.py')
    parser.add_argument('save_dir', type=str, help='where to save files.')
    args: Any = parser.parse_args()

    main(args)
