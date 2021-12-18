"""
For crazy speed reasons, we can't afford to cast to 
set in real time. So let's do it once.
"""

import pickle
from tqdm import tqdm
from typing import Any, Dict, List, Set


def main(args: Any):
    data_new: Dict[int, Set[int]] = {}

    with open(args.edges_raw_file, 'rb') as fp:
        data: Dict[int, List[int]] = pickle.load(fp)

        pbar = tqdm(total=len(data))
        for node in data.keys():
            edges: List[int] = data[node]
            edges: Set[int] = set(edges) - {node}
            data_new[node] = edges
            pbar.update()
        pbar.close()

    with open(args.out_file, 'wb') as fp:
        pickle.dump(data_new, fp)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('edges_raw_file', type=str, help='path to load data')
    parser.add_argument('out_file', type=str, help='path to save data')
    args: Any = parser.parse_args()

    main(args)

