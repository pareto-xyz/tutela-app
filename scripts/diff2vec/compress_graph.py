"""
Rather than storing addresses in DataFrames, store integers (much smaller). 
Then we can we create a map to do the replacement later.
"""

import numpy as np
import pandas as pd
from typing import Any, Iterable, Dict, Set, List

from src.utils.utils import to_json


def yield_transactions(
    transactions_csv: str, chunk_size: int = 10000) -> Iterable[pd.DataFrame]:
    """
    Load a segment at a time (otherwise too large).
    """
    for chunk in pd.read_csv(transactions_csv, chunksize = chunk_size):
        yield chunk


def make_address_map(
    address_map: Dict[str, int], 
    chunk: pd.DataFrame, 
    min_index: int,
) -> Dict[str, Any]:
    from_set: Set[str] = set(chunk.from_address.to_numpy())
    to_set: Set[str] = set(chunk.to_address.to_numpy())

    # build mapping from unique addresses
    addresses: Set[str] = from_set.union(to_set)
    bank: Set[str] = set(list(address_map.keys()))
    # remove existing addresses
    addresses: Set[str] = addresses - bank
    addresses: List[str] = sorted(list(addresses))

    indices: List[int] = list(range(len(addresses)))
    indices: List[int] = [x + min_index for x in indices]

    return dict(zip(addresses, indices))


def make_graph_dataframe(
    transactions_csv: str,
    out_csv: str,
    addr_json: str,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    count: int = 0
    address_map: Dict[str, int] = {}

    print('processing txs',  end = '', flush=True)
    for chunk in yield_transactions(transactions_csv, chunk_size):
        cur_map: Dict[str, int] = \
            make_address_map(address_map, chunk, len(address_map))
        address_map.update(cur_map)

        chunk.from_address = chunk.from_address.apply(lambda x: address_map[x])
        chunk.to_address = chunk.to_address.apply(lambda x: address_map[x])

        if count == 0:
            chunk.to_csv(out_csv, index=False)
        else:
            chunk.to_csv(out_csv, mode='a', header=False, index=False)

        del chunk  # wipe memory
        print('.', end = '', flush=True)
        count += 1 

    to_json(address_map, addr_json)


def main(args: Any):
    make_graph_dataframe(
        args.transactions_csv, args.save_csv, args.addr_json, args.chunk_size)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('transactions_csv', type=str, help='path to transaction data')
    parser.add_argument('save_csv', type=str, help='path to save data')
    parser.add_argument('addr_json', type=str, help='path to save addr map')
    parser.add_argument('--chunk-size', type=int, default=1000000,
                        help='Chunk size (default: 1000000)')
    args: Any = parser.parse_args()

    main(args)
