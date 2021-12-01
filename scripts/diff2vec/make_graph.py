"""
Need to convert the transactions dataframe to a smaller dataframe 
with the columns: `Address A | Address B | # of interactions`
"""
import pandas as pd
from typing import Any, Iterable


def yield_transactions(
    transaction_csv: str,
    chunk_size: int = 10000,
) -> Iterable[pd.DataFrame]:
    """
    Load a segment at a time (otherwise too large).
    """
    for chunk in pd.read_csv(transaction_csv, chunksize = chunk_size):
        yield chunk


def make_graph_dataframe(
    transaction_csv: str,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    for chunk in yield_transactions(transaction_csv, chunk_size):
        chunk.value = chunk.value.astype(float) / 10**18


def main(args: Any):
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('transactions_csv', type=str, help='path to transaction data')
    parser.add_argument('save_dir', type=str, help='path to save output')
    args: Any = parser.parse_args()

    main(args)
