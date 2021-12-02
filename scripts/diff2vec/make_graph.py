"""
Need to convert the transactions dataframe to a smaller dataframe 
with the columns: `Address A | Address B | # of interactions`
"""
import numpy as np
import pandas as pd
from typing import Any, Iterable


def yield_transactions(
    transactions_csv: str, chunk_size: int = 10000) -> Iterable[pd.DataFrame]:
    """
    Load a segment at a time (otherwise too large).
    """
    for chunk in pd.read_csv(transactions_csv, chunksize = chunk_size):
        yield chunk


def make_graph_dataframe(
    transactions_csv: str, out_csv: str, chunk_size: int = 10000) -> pd.DataFrame:
    count: int = 0

    print('processing txs',  end = '', flush=True)
    for chunk in yield_transactions(transactions_csv, chunk_size):
        chunk: pd.DataFrame = \
            chunk.groupby(['from_address', 'to_address'], as_index=False).size()

        if count == 0:
            chunk.to_csv(out_csv, index=False)
        else:
            chunk.to_csv(out_csv, mode='a', header=False, index=False)

        print('.', end = '', flush=True)
        count += 1 


def main(args: Any):
    make_graph_dataframe(
        args.transactions_csv, args.save_csv, args.chunk_size)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('transactions_csv', type=str, help='path to transaction data')
    parser.add_argument('save_csv', type=str, help='path to save data')
    parser.add_argument('--chunk-size', type=int, default=1000000,
                        help='Chunk size (default: 1000000)')
    args: Any = parser.parse_args()

    main(args)
