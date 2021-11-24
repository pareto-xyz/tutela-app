"""
The file `data.csv` is produced by running `src/cluster/deposit.py` but these
results could noisy. We apply a set of post-processing rules to at least ensure
consistency. 

RULE #1: It is possible for A -> B -> Ex, and C -> A -> Ex to both appear. We 
don't want to consider `A` an eoa in one setting, and a deposit in another setting.
It is very unlikely for `A` to be a desposit if we see it do `A -> B -> Ex`. Delete
these entries from `data.csv`.

RULE #2: We are always certain about exchange addresses and so any lines with 
them as EOAs or deposits can be removed.
"""
import numpy as np
import pandas as pd
from typing import Any


def main(args: Any):
    df: pd.DataFrame = pd.read_csv(args.data_csv)
    exchanges: np.array = df.exchange.unique()
    print(f'init: {len(df)} rows.')

    # Exchanges cannot be users or deposits
    df = df[~df.user.isin(exchanges)]
    df = df[~df.deposit.isin(exchanges)]
    print(f'after removing exchanges as eoa/deposits: {len(df)} rows.')

    # Find all users and make sure they cannot be deposits since 
    # deposits cannot send to A -> B -> Exchange.
    users: np.array = df.user.unique()
    df = df[~df.deposit.isin(users)]
    print(f'after removing deposits who are also eoa\'s: {len(df)} rows.')
    
    print('saving to file...')
    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_csv', type=str)
    parser.add_argument('out_csv', type=str)
    args = parser.parse_args()

    main(args)

