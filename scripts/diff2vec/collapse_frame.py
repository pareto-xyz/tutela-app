"""
The script `make_graph.py` writes greedily in that it has 
duplicates, which could greatly affect size. Write a script
that iterates through blocks of some size and can be repeatedly
called to compress the size.
"""

import pandas as pd
from typing import Any


def main(args: Any):
    start: int = 0
    size: int = 1

    while size > 0:
        df: pd.DataFrame = pd.read_csv(
            args.raw_csv,
            nrows = args.chunk_size,
            skiprows = start,
        )
        df: pd.DataFrame = \
            df.groupby(['from_address', 'to_address'], as_index=False).sum()

        if start == 0:
            df.to_csv(args.out_csv, index=False)
        else:
            df.to_csv(args.out_csv, mode='a', header=False, index=False)
        
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
