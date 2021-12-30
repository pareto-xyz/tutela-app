"""
Process same gas price clusters to CSV (same table format).
"""

import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple


def from_json(path):
    with open(path, 'r') as fp:
        return json.load(fp)


def main(args: Any):
    clusters: List[Set[str]] = from_json(args.clusters_file)
    tx2addr: Dict[str, str] = from_json(args.tx2addr_file)
    tx2block: Dict[str, int] = from_json(args.tx2block_file)
    tx2ts: Dict[str, Any] = from_json(args.tx2ts_file)

    transactions, tx2cluster = get_transactions(clusters)
    transactions: List[str] = list(transactions)
    addresses: List[str] = [tx2addr[tx] for tx in transactions]
    block_numbers: List[int] = [tx2block[tx] for tx in transactions]
    block_timestamps: List[Any] = [tx2ts[tx] for tx in transactions]
    block_timestamps: List[datetime] = [
        datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in block_timestamps]
    clusters: List[int] = [tx2cluster[tx] for tx in transactions]
    meta_datas:  List[str] = [json.dumps({}) for _ in transactions]

    dataset: Dict[str, List[Any]] = {
        'address': addresses,
        'transaction': transactions,
        'block_number': block_numbers,
        'block_ts': block_timestamps,
        'meta_data': meta_datas,
        'cluster': clusters,
    }
    df: pd.DataFrame = pd.DataFrame.from_dict(dataset)
    df.to_csv(args.out_file, index=False)


def get_transactions(
    clusters: List[Set[str]],
) -> Tuple[Set[str], Dict[str, int]]:
    transactions: Set[str] = set()
    tx2cluster: Dict[str, int] = {}
    pbar = tqdm(total=len(clusters))
    for c, cluster in enumerate(clusters):
        transactions = transactions.union(cluster)
        for tx in cluster:  # put all 
            tx2cluster[tx] = c
        pbar.update()
    pbar.close()

    return transactions, tx2cluster


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('clusters_file', type=str)
    parser.add_argument('tx2addr_file', type=str)
    parser.add_argument('tx2block_file', type=str)
    parser.add_argument('tx2ts_file', type=str)
    parser.add_argument('out_file', type=str)
    args = parser.parse_args()

    main(args)
