import itertools
import numpy as np
import pandas as pd
from typing import Any, List, Tuple

from src.diff2vec.graph import UndirectedGraph
# from src.diff2vec.diff2vec import Diff2Vec


def main(args: Any):
    data: pd.DataFrame = pd.read_csv(args.data_csv)
    graph: UndirectedGraph = build_graph(data)

    # model: Diff2Vec = Diff2Vec(
    #     dimensions = args.dim,
    #     window_size = args.window,
    #     cover_size = args.cover,
    #     epochs = args.epochs,
    #     learning_rate = args.lr,
    #     workers = args.workers,
    #     seed = args.seed,
    # )
    # model.fit(graph)


def build_graph(data: pd.DataFrame) -> UndirectedGraph:
    node_a: List[int] = data.from_address.to_numpy().tolist()
    node_b: List[int] = data.to_address.to_numpy().tolist()
    edge_ab: List[Tuple[int, int]] = list(itertools.product(node_a, node_b))

    graph: UndirectedGraph = UndirectedGraph()
    graph.add_nodes_from(node_a)
    graph.add_nodes_from(node_b)
    graph.add_edges_from(edge_ab)

    return graph



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_csv', type=str, help='path to save data')
    parser.add_argument('--epochs', type=int, default=10, help='epochs (default: 10)')
    parser.add_argument('--workers', type=int, default=4, help='workers (default: 4)')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate (default: 0.05)')
    parser.add_argument('--dim', type=float, default=128, help='dimensionality (default: 128)')
    parser.add_argument('--window', type=float, default=10, help='window (default: 10)')
    parser.add_argument('--cover', type=float, default=80, help='cover (default: 80)')
    parser.add_argument('--seed', type=float, default=42, help='random seed (default: 42)')
    args: Any = parser.parse_args()

    main(args)
