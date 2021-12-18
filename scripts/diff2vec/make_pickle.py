"""
The graph is huge: we need to break it up into chunks.
"""
import os
import pandas as pd
from typing import Any, List, Tuple
from src.diff2vec.graph import UndirectedGraph


def main(args: Any):
    print('Loading data from CSV')
    data: pd.DataFrame = pd.read_csv(args.data_csv)
    print('Building graph')
    graph: UndirectedGraph = build_graph(data)
    print(f'Made graph with {len(graph)} nodes')

    edge_file: str = os.path.join(args.cache_dir, f'edges-raw.pickle')
    graph.to_pickle(edge_file)


def build_graph(data: pd.DataFrame) -> UndirectedGraph:
    node_a: List[int] = data.from_address.to_numpy().tolist()
    node_b: List[int] = data.to_address.to_numpy().tolist()
    edge_ab: List[Tuple[int, int]] = list(zip(node_a, node_b))

    graph: UndirectedGraph = UndirectedGraph()
    graph.add_nodes_from(node_a)
    graph.add_nodes_from(node_b)
    graph.add_edges_from(edge_ab)

    return graph


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_csv', type=str, help='path to save data')
    parser.add_argument('cache_dir', type=str, help='path to cache')
    args: Any = parser.parse_args()

    main(args)
