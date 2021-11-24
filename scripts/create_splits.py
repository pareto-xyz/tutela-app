"""
Once we have all the block data, create splits of 

- 1 week    (toy debugging set)
- 1 month   (cheaper test set)
- 1 year    (more expensive test set)
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Any


def main(args: Any):
    root_1week: str = os.path.join(
        args.root, 'processed', 'blocks-1week.csv')
    root_1month: str = os.path.join(
        args.root, 'processed', 'blocks-1month.csv')
    root_1year: str = os.path.join(
        args.root, 'processed', 'blocks-1year.csv')
    root_all: str  = os.path.join(
        args.root, 'processed', 'blocks-sorted.csv')

    # start of the month that we scrapped at
    end_date: str = '2021-10-01 00:00:00 UTC'
    end_date: datetime = datetime.strptime(
        end_date, 
        '%Y-%m-%d %H:%M:%S UTC',
    )

    start_date_1week: datetime = end_date - timedelta(days=7)
    start_date_1month: datetime = end_date - timedelta(days=30)
    start_date_1year: datetime = end_date - timedelta(days=365)

    print('Reading large file...')
    df = pd.read_csv(root_all, index_col=[0])
    breakpoint()

    df.timestamp = pd.to_datetime(
        df.timestamp,
        format='%Y-%m-%d %H:%M:%S UTC',
    )
    slice_1week: pd.Series = np.logical_and(
        df.timestamp >= start_date_1week,
        df.timestamp <= end_date,
    )
    slice_1month: pd.Series = np.logical_and(
        df.timestamp >= start_date_1month,
        df.timestamp <= end_date,
    )
    slice_1year: pd.Series = np.logical_and(
        df.timestamp >= start_date_1year,
        df.timestamp <= end_date,
    )

    df_1week: pd.DataFrame = df[slice_1week]
    df_1month: pd.DataFrame = df[slice_1month]
    df_1year: pd.DataFrame = df[slice_1year]

    print('Saving splits...')
    df_1week.to_csv(root_1week)
    df_1month.to_csv(root_1month)
    df_1year.to_csv(root_1year)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
        default='./data/bigquery/ethereum-block-data', 
        help='path to data root (default: ./data/bigquery/ethereum-block-data)',
    )
    args: Any = parser.parse_args()

    main(args)
