import os
import jsonlines
import pandas as pd
from typing import Any, List, Set

from src.diff2vec.graph import UndirectedGraphCSV


def main(args: Any):
    graph: UndirectedGraphCSV = UndirectedGraphCSV(args.edges_dir)
    graph.connected_components(args.out_file)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('edges_dir', type=str, help='path split edges file')
    parser.add_argument('out_file', type=str, help='path to save components')
    args: Any = parser.parse_args()

    main(args)
