"""
As the metadata is stored, deposit addresses might be associated with
multiple exchanges. EOA addresses might appear multiple times as well.

We stored the metadata in chunks; as such there may be duplicates. Let
us just take the most confident of each.

Input: data.csv
Output: data-pruned.csv
"""

import pandas as pd
from typing import Any


def main(args: Any):
    df: pd.DataFrame = pd.read_csv(args.metadata_csv)
    print(f'init: {len(df)} rows.')
    print('Running large groupby job...')
    pruned_df: pd.DataFrame = df.loc[df.groupby(['address'])['conf'].idxmax()]
    print(f'after pruning: {len(pruned_df)} rows.')
    pruned_df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata_csv', type=str)
    parser.add_argument('out_csv', type=str)
    args = parser.parse_args()

    main(args)
