import os
from web3 import Web3
import pandas as pd
from typing import Any, Dict


def main(args: Any):
    withdraw_file: str = os.path.join(args.data_dir, 'withdraw_transactions.csv')
    abi_file: str = os.path.join(args.data_dir, 'tornadocontracts_abi.csv')
    withdraw_df: pd.DataFrame = pd.read_csv(withdraw_file)
    addresses_df: pd.DataFrame = pd.read_csv(
        abi_file,
        names=['address', 'token', 'value', 'name', 'abi'],
        sep='|',
    )
    contracts: pd.DataFrame = get_contracts(addresses_df)
    withdraw_df['recipient_address'] = withdraw_df.apply(
        lambda row: recipient_address(row, contracts), axis=1)

    out_path: str = os.path.join(args.data_dir, 'tornado_withdraw_df.csv')
    withdraw_df.to_csv(out_path)


def get_contracts(data: pd.DataFrame):
    # creates contract object given it address and abi
    w3: Web3 = Web3(Web3.HTTPProvider('https://cloudflare-eth.com'))
    contracts: Dict[str, Any] = {}
    for _, row in data.iterrows():
        address: str = row['address']
        contracts[address] = w3.eth.contract(
            address=w3.toChecksumAddress(address), 
            abi=eval(row['abi']),
        )
    return contracts


def recipient_address(row: pd.Series, contracts: Dict[str, Any]):
    to_address: str = row['to_address']
    input_data: str = row['input']
    _, func_params = contracts[to_address].decode_function_input(input_data)

    return func_params["_recipient"]


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to tornado cash data')
    args: Any = parser.parse_args()

    main(args)