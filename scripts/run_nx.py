"""
To scale the run_deposit.py script, we had to forgo creating 
NX graph in memory. This script does exactly that. The motivation
is to isolate the high memory parts to a single file.
"""

import os
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from src.utils.utils import to_json, from_json
from typing import Any, List, Set, Tuple


def main(args: Any):
    data: pd.DataFrame = pd.read_csv(args.data_file)

    gas_price_sets: List[Set[str]] = from_json(args.gas_price_file)
    multi_denom_sets: List[Set[str]] = from_json(args.multi_denom_file)

    print('making user graph...',  end = '', flush=True)
    user_graph: nx.DiGraph = make_graph(data.user, data.deposit)

    print('adding gas price nodes...', end = '', flush=True)
    user_graph: nx.DiGraph = add_to_user_graph(user_graph, gas_price_sets)

    print('adding multi denom nodes...', end = '', flush=True)
    user_graph: nx.DiGraph = add_to_user_graph(user_graph, multi_denom_sets)

    print('making exchange graph...',  end = '', flush=True)
    exchange_graph: nx.DiGraph = make_graph(data.deposit, data.exchange)

    print('making user wcc...',  end = '', flush=True)
    user_wccs: List[Set[str]] = get_wcc(user_graph)

    # algorithm 1 line 13
    # We actually want to keep this information!
    # user_wccs: List[Set[str]] = self._remove_deposits(
    #     user_wccs,
    #     set(store.deposit.to_numpy().tolist()),
    # )

    print('making exchange wcc...',  end = '', flush=True)
    exchange_wccs: List[Set[str]] = get_wcc(exchange_graph)

    # prune trivial clusters
    user_wccs: List[Set[str]] = remove_singletons(user_wccs)
    exchange_wccs: List[Set[str]] = remove_singletons(exchange_wccs)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    print('writing to disk...\n',  end = '', flush=True)
    to_json(user_wccs, os.path.join(args.save_dir, 'user_clusters.json'))
    to_json(exchange_wccs, os.path.join(args.save_dir, 'exchange_clusters.json'))


def add_to_user_graph(graph: nx.DiGraph, clusters: List[Set[str]]):
    for cluster in clusters:
        assert len(cluster) == 2, "Only supports edges with two nodes."
        node_a, node_b = cluster
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_edge(node_a, node_b)
    return graph


def get_wcc(graph: nx.DiGraph) -> List[Set[str]]:
    comp_iter: Any = nx.weakly_connected_components(graph)
    comps: List[Set[str]] = [c for c in comp_iter]
    return comps

def remove_deposits(components: List[Set[str]], deposit: Set[str]):
    # remove all deposit addresses from wcc list
    new_components: List[Set[str]] = []
    for component in components:
        new_component: Set[str] = component - deposit
        new_components.append(new_component)

    return new_components


def remove_singletons(components: List[Set[str]]):
    # remove clusters with just one entity... these are not interesting.
    return [c for c in components if len(c) > 1]


def make_graph(node_a: pd.Series, node_b: pd.Series) -> nx.DiGraph:
    """
    DEPRECATED: This assumes we can store all connections in memory.

    Make a directed graph connecting each row of node_a to the 
    corresponding row of node_b.
    """
    assert node_a.size == node_b.size, "Dataframes are uneven sizes."

    graph: nx.DiGraph = nx.DiGraph()

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
    parser.add_argument('gas_price_file', type=str, help='path to gas price address sets')
    parser.add_argument('multi_denom_file', type=str, help='path to gas price address sets')
    parser.add_argument('save_dir', type=str, help='where to save files.')
    args: Any = parser.parse_args()

    main(args)
