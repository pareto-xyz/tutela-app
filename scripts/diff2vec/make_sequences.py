import os
import h5py
import pandas as pd
from typing import Any, List, Set

from src.diff2vec.graph import UndirectedGraphCSV
from src.diff2vec.euler import SubGraphSequences


def main(args: Any):
    # sequence_file: str = os.path.join(
    #     args.cache_dir, f'sequences-{args.cover_size}.jsonl')

    graph: UndirectedGraphCSV = UndirectedGraphCSV(args.edges_dir)

    components: List[Set[str]] = graph.connected_components()
    # print('Computing subgraph sequences')
    # sequencer: SubGraphSequences = SubGraphSequences(graph, args.cover_size)
    # sequencer.get_sequences(sequence_file)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('edges_dir', type=str, help='path split edges file')
    # parser.add_argument('cache_dir', type=str, help='path to cache')
    parser.add_argument('--cover-size', type=int, default=80, 
                        help='size of subgraph (default: 80)')
    args: Any = parser.parse_args()

    main(args)
