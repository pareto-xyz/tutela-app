import os
import h5py
import pandas as pd
from typing import Any, List, Set

from src.diff2vec.graph import UndirectedGraphCSV
from src.diff2vec.euler import SubGraphSequences


def main(args: Any):
    graph: UndirectedGraphCSV = UndirectedGraphCSV(args.edges_dir)
    sequencer: SubGraphSequences = SubGraphSequences(graph, args.cover_size)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('edges_dir', type=str, help='path split edges file')
    parser.add_argument('--cover-size', type=int, default=80, 
                        help='size of subgraph (default: 80)')
    args: Any = parser.parse_args()

    main(args)
