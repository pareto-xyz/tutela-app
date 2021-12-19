import os
from typing import Any

from src.diff2vec.graph import UndirectedGraph
from src.diff2vec.euler import SubGraphSequences


def main(args: Any):
    graph: UndirectedGraph = UndirectedGraph()
    graph.from_pickle(args.edges_file)
    sequencer: SubGraphSequences = SubGraphSequences(graph, args.cover_size)

    for i in range(args.replicates):
        name, ext = os.path.splitext(args.sequences_file)
        sequences_file: str = f'{name}_rep{i}' + ext
        sequencer.get_sequences(args.components_file, sequences_file)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('edges_file', type=str, help='path to edges pickle file')
    parser.add_argument('components_file', type=str, help='path to components file')
    parser.add_argument('sequences_file', type=str, help='path to sequences file')
    parser.add_argument('--cover-size', type=int, default=80, 
                        help='size of subgraph (default: 80)')
    parser.add_argument('--replicates', type=int, default=1,
                        help='number of replicates (default: 1)')
    args: Any = parser.parse_args()

    main(args)
