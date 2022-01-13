"""
Get a random sample of addresses, compute a distribution over number of DAR 
transaction reveals. Get a random sample of Tornado Cash users (those who 
have deposited), compute a distribution over number of tornado specific reveals.
"""

import os
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import Dict, List, Any
from app.models import ExactMatch, GasPrice, MultiDenom, LinkedTransaction, \
                       TornMining, TornadoDeposit, DepositTransaction
from app.utils import GAS_PRICE_HEUR, DEPO_REUSE_HEUR, SAME_NUM_TX_HEUR, \
                      SAME_ADDR_HEUR, LINKED_TX_HEUR, TORN_MINE_HEUR
from src.utils.utils import to_json, to_pickle

HEURISTICS: List[Any] = [ExactMatch, GasPrice, MultiDenom, 
                         LinkedTransaction, TornMining]
NAMES: List[Any] = [SAME_ADDR_HEUR, GAS_PRICE_HEUR, SAME_NUM_TX_HEUR,
                    LINKED_TX_HEUR, TORN_MINE_HEUR]


def get_tornado_cash_users(size: int, rs: np.random.RandomState) -> List[str]:
    rows: List[TornadoDeposit] = TornadoDeposit.query.all()
    addresses: List[str] = [row.from_address for row in rows]
    addresses: np.array = np.array(addresses)
    addresses: np.array = np.unique(addresses)

    if size <= 0: return []
    if size > len(addresses): return addresses.tolist()
    addresses: np.array = rs.choice(addresses, size, replace=False)
    return addresses.tolist()


def get_score_dist(addresses: List[str]) -> Dict[str, Dict[int, int]]:
    scores: Dict[str, List[str]] = {}
    
    scores[DEPO_REUSE_HEUR] = []
    for name in NAMES: 
        scores[name] = []

    pbar = tqdm(total=len(addresses))
    for address in addresses:
        num: int = find_num_dar_matches(address)
        scores[DEPO_REUSE_HEUR].append(num)
        for name, heuristic in zip(NAMES, HEURISTICS):
            num: int = find_num_tcash_matches(address, heuristic)
            scores[name].append(num)
        pbar.update()
    pbar.close()

    dist: Dict[str, Dict[int, int]] = {}
    dist[DEPO_REUSE_HEUR] = dict(Counter(scores[DEPO_REUSE_HEUR]))
    for name in NAMES:
        dist[name] =  dict(Counter(scores[name]))
    return dist


def find_num_tcash_matches(address: str, Heuristic: Any) -> int:
    rows: List[Heuristic] = \
        Heuristic.query.filter(Heuristic.address == address).all()
    return len(rows)


def find_num_dar_matches(address: str) -> int:
    rows: List[DepositTransaction] = \
        DepositTransaction.query.filter(DepositTransaction.address == address).all()
    return len(rows)


def main(args: Any):
    rs: np.random.RandomState = np.random.RandomState(args.seed)
    addresses: List[str] = get_tornado_cash_users(args.size, rs)
    score_dists: Dict[str, Dict[int, int]] = get_score_dist(addresses)

    out_file: str = os.path.join(args.data_dir, 'transaction_reveal_dist.pickle')
    to_pickle(score_dists, out_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='where to save output')
    parser.add_argument('--size', type=int, default=100000, 
                        help='number of samples to estimate distribution (default: 100000)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    args = parser.parse_args()

    main(args)
