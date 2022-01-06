"""
Create a dataset of complete Tcash transactions.

https://github.com/lambdaclass/tornado_cash_anonymity_tool/blob/main/notebooks/complete_dataset.ipynb
"""

import os
import pandas as pd
from typing import Any
from src.tcash.data import decode_transactions


def main(args: Any):
    trace_df: pd.DataFrame = pd.read_csv(
        os.path.join(args.data_dir, 'tornado_traces.csv'), low_memory=False)
    transaction_df: pd.DataFrame = pd.read_csv(
        os.path.join(args.data_dir, 'tornado_transactions.csv'))
    address_df: pd.DataFrame = pd.read_csv(
        os.path.join(args.contract_dir, 'tornado_contract_abi.csv'),
        names=['address', 'token', 'value', 'name','abi'],
        sep='|')
    proxy_df = pd.read_csv(
        os.path.join(args.contract_dir, 'tornado_proxy_abi.csv'), 
        names=['address', 'abi'],
        sep='|')

    deposit_df, withdraw_df = decode_transactions(
        address_df, proxy_df, transaction_df, trace_df)
    withdraw_df.to_csv(
        os.path.join(args.data_dir, 'complete_withdraw_txs.csv'), index=False)
    deposit_df.to_csv(
        os.path.join(args.data_dir, 'complete_deposit_txs.csv'), index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to trace and transaction data')
    parser.add_argument('contract_dir', type=str, help='path to contract data')
    args: Any = parser.parse_args()
    main(args)
