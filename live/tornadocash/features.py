"""
This script will be run after data.py and heuristics.py.

This assumes access to database and that it is already loaded.

Currently, this computes distribution statistics for the 
transactions page.

Additional features should be added to this page.
"""
import sys
import numpy as np
from os.path import join
from typing import Any, List, Dict

from live import utils
from src.utils.utils import to_pickle

sys.path.append(utils.CONSTANTS['webapp_path'])
from db_utils.get_dist import get_tornado_cash_users, get_score_dist


def main(args: Any):
    rs: np.random.RandomState = np.random.RandomState(args.seed)
    addresses: List[str] = get_tornado_cash_users(args.size, rs)
    score_dists: Dict[str, Dict[int, int]] = get_score_dist(addresses)

    data_path:  str = utils.CONSTANTS['webapp_data_path']
    out_file: str = join(data_path, 'transaction_reveal_dist.pickle')
    to_pickle(score_dists, out_file)  # update raw file!


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=100000, 
                        help='number of samples to estimate distribution (default: 100000)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--no-db', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
