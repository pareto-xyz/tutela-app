import os
from typing import Any

from src.diff2vec.graph import UndirectedGraph
from src.diff2vec.euler import SubGraphSequences


def main(args: Any):
    graph: UndirectedGraph = UndirectedGraph(args.edges_file)
    sequencer: SubGraphSequences = SubGraphSequences(graph, args.cover_size)
    sequencer.get_sequences()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('edges_file', type=str, help='path to edges pickle file')
    parser.add_argument('edges_file', type=str, help='path to edges pickle file')
    parser.add_argument('--cover-size', type=int, default=80, 
                        help='size of subgraph (default: 80)')
    args: Any = parser.parse_args()

    main(args)
