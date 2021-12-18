from typing import Any
from src.diff2vec.graph import UndirectedGraph


def main(args: Any):
    graph: UndirectedGraph = UndirectedGraph()
    graph.from_pickle(args.edges_file)
    breakpoint()
    graph.connected_components(args.out_file)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('edges_file', type=str, help='path to edges pickle file')
    parser.add_argument('out_file', type=str, help='path to save components')
    args: Any = parser.parse_args()

    main(args)
