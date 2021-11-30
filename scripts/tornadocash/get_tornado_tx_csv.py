"""
Create a CSV that can be uploaded directly into TornadoPool table. 
Looks over all interactions and records which interactions are depositing
into which tornado pools?
"""
import pandas as pd
from typing import Any


def main(args: Any):
    deposits: pd.DataFrame = pd.read_csv(args.deposit_csv)
    deposits: pd.DataFrame = deposits[['hash', 'from_address', 'tornado_cash_address']]
    deposits.rename(columns={
        'hash': 'transaction', 
        'from_address': 'address', 
        'tornado_cash_address': 'pool'}, inplace=True)
    deposits.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('deposit_csv', type=str, help='path to tornado cash deposit data')
    parser.add_argument('out_csv', type=str, help='where to save output file?')
    args: Any = parser.parse_args()

    main(args)
