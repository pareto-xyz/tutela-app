from typing import Any
from src.diff2vec.graph import UndirectedGraphCSV, UndirectedGraphCSVSplits


def main(args: Any):
    if args.use_splits:
        graph: UndirectedGraphCSVSplits = UndirectedGraphCSVSplits(args.edges)
    else:
        graph: UndirectedGraphCSV = UndirectedGraphCSV(args.edges)
    graph.connected_components(args.out_file)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('edges', type=str')
    parser.add_argument('out_file', type=str, help='path to save components')
    parser.add_argument('--use-splits', action='store_true', default=False, 
                        help='use splits')
    args: Any = parser.parse_args()

    main(args)
