"""
Need to convert the transactions dataframe to a smaller dataframe 
with the columns: `Address A | Address B | # of interactions`
"""
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
    transactions_csv: str, 
    known_addresses_csv: str,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    # we want to ignore these addresses as they are not interesting to 
    # cluster against.
    known_addresses: pd.DataFrame = pd.read_csv(known_addresses_csv)

    for chunk in yield_transactions(transactions_csv, chunk_size):
        breakpoint()


def main(args: Any):
    data: pd.DataFrame = make_graph_dataframe(
        args.transactions_csv, args.known_addresses_csv, args.chunk_size)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('transactions_csv', type=str, help='path to transaction data')
    parser.add_argument('known_addresses_csv', type=str, help='path to known address data')
    parser.add_argument('save_dir', type=str, help='path to save output')
    parser.add_argument('--chunk-size', type=int, help='Chunk size (default: 10000)')
    args: Any = parser.parse_args()

    main(args)
