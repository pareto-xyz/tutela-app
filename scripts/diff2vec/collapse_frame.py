"""
The script `make_graph.py` writes greedily in that it has 
duplicates, which could greatly affect size. Write a script
that iterates through blocks of some size and can be repeatedly
called to compress the size.
"""

import pandas as pd
from typing import Any


def main(args: Any):
    start: int = 1  # skip header
    size: int = 1

    print('collapsing entries',  end = '', flush=True)
    while size > 0:
        df: pd.DataFrame = pd.read_csv(
            args.raw_csv,
            nrows = args.chunk_size,
            skiprows = start,
            header = None,
        )
        df.columns = ['from_address', 'to_address', 'size']  # manually name columns
        subset: pd.DataFrame = \
            df.groupby(['from_address', 'to_address'], as_index=False).sum()

        if start == 0:
            subset.to_csv(args.out_csv, index=False)
        else:
            subset.to_csv(args.out_csv, mode='a', header=False, index=False)
        print('.', end = '', flush=True)
        size: int = len(df)
        start += size


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('raw_csv', type=str, help='path to raw df csv')
    parser.add_argument('out_csv', type=str, help='path to out csv')
    parser.add_argument('--chunk-size', type=int, default=1000000,
                        help='Chunk size (default: 1000000)')
    args: Any = parser.parse_args()

    main(args)