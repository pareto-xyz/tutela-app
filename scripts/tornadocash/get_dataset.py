"""
Create a dataset of complete Tcash transactions.

https://github.com/lambdaclass/tornado_cash_anonymity_tool/blob/main/notebooks/complete_dataset.ipynb
"""

import os
import pandas as pd
from web3 import Web3
from tqdm import tqdm
from typing import Any, Dict, Tuple


def get_tornado_contracts(tornado_df) -> Dict[str, Any]:
    """
    Creates contract object given it address and abi.
    """
    w3: Web3 = Web3(Web3.HTTPProvider('https://cloudflare-eth.com'))
    contracts: Dict[str, Any] = {}
    for _, row in tornado_df.iterrows():
        address: str = row['address']
        contracts[address] = w3.eth.contract(
            address=w3.toChecksumAddress(address), 
            abi=eval(row['abi']),
        )
    return contracts


def decode_transactions(
    contract_df: pd.DataFrame, 
    proxy_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    trace_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decode the input data from those transaction (the ones going through 
    the TCash proxy) to be able to know the pool (Tcash contract) that will 
    be interacting in the internal transaction.

    As there is the possibility of making a withdrawal through a relayer, 
    the information about the wallet that is making the withdrawal is not 
    stored in the field "from_address" (in the field "to_address" there are 
    always the addresses of the tornado_cash contracts), but it is stored 
    as one of the parameters of the withdrawal function, which is coded in 
    the field "input". This function decodes information stored in the input.
    """
    # split traces into deposits and withdraws
    deposit_trace_df: pd.DataFrame = trace_df[trace_df['input'].str[:10] == '0xb214faa5']
    withdraw_trace_df: pd.DataFrame = trace_df[trace_df['input'].str[:10] == '0x21a0adb6']

    # find corresponding transactions
    deposit_transaction_df = transaction_df[transaction_df['hash']\
        .isin(deposit_trace_df['transaction_hash'])]
    withdraw_transaction_df = transaction_df[transaction_df['hash']\
        .isin(withdraw_trace_df['transaction_hash'])]

    # sanity checks
    assert deposit_trace_df.shape[0] + withdraw_trace_df.shape[0] == trace_df.shape[0]
    assert deposit_transaction_df.shape[0] + withdraw_transaction_df.shape[0] == transaction_df.shape[0]

    # filtering the deposits by the ones that have the Tcash proxy address:
    # 0x722122df12d4e14e13ac3b6895a86e84145b6967 is for the current Tcash proxy
    # 0x905b63fff465b9ffbf41dea908ceb12478ec7601 is the address of the "old proxy"
    proxy_deposit_df = deposit_transaction_df[deposit_transaction_df['to_address']\
        .isin(['0x722122df12d4e14e13ac3b6895a86e84145b6967', 
                '0x905b63fff465b9ffbf41dea908ceb12478ec7601'])]

    # use web3 in order to decode the input data and get the pool that the 
    # proxy is going to interact with
    true = True; false = False  # contract abi

    proxy_contracts: Dict[str, Any] = get_tornado_contracts(proxy_df)
    contracts: Dict[str, Any] = get_tornado_contracts(contract_df)

    def get_tornado_address(row: pd.Series, contracts: Dict[str, Any]):
        _, func_params = contracts[row['to_address']].decode_function_input(row['input'])
        return func_params['_tornado'].lower()

    def get_tornado_recipient(row: pd.Series, contracts: Dict[str, Any]):
        _, func_params = contracts[row['to_address']].decode_function_input(row['input'])
        return func_params['_recipient'].lower()

    # Adds the column with the Tcash contracts that the proxy is going to 
    # interact with (in a internal tx)
    proxy_deposit_df['tornado_cash_address'] = proxy_deposit_df.apply(
        lambda row: get_tornado_address(row, proxy_contracts), axis=1)

    # add the same column to the tx that goes from a wallet directly to a TCash contract
    tcash_deposit_df = deposit_transaction_df[
        deposit_transaction_df['to_address'].isin(contract_df['address'])]
    tcash_deposit_df['tornado_cash_address'] = tcash_deposit_df['to_address']

    complete_deposit_df: pd.DataFrame = pd.concat([tcash_deposit_df, proxy_deposit_df])

    # do the same for withdraws 
    proxy_withdraw_df = withdraw_transaction_df[withdraw_transaction_df['to_address']\
        .isin(['0x722122df12d4e14e13ac3b6895a86e84145b6967', 
                '0x905b63fff465b9ffbf41dea908ceb12478ec7601'])]
    tcash_withdraw_df = withdraw_transaction_df[
        withdraw_transaction_df['to_address'].isin(contract_df['address'])]
    
    proxy_withdraw_df['tornado_cash_address'] = proxy_withdraw_df.apply(
        lambda row: get_tornado_address(row, proxy_contracts), axis=1)

    proxy_withdraw_df['recipient_address'] = proxy_withdraw_df.apply(
        lambda row: get_tornado_recipient(row, proxy_contracts), axis=1)

    tcash_withdraw_df['tornado_cash_address'] = tcash_withdraw_df['to_address']
    tcash_withdraw_df['recipient_address'] = tcash_withdraw_df.apply(
        lambda row: get_tornado_recipient(row, contracts), axis=1)

    complete_withdraw_df: pd.DataFrame = pd.concat([tcash_withdraw_df, proxy_withdraw_df])

    return complete_deposit_df, complete_withdraw_df


def main(args: Any):
    trace_df: pd.DataFrame = pd.read_csv(
        os.path.join(args.data_dir, 'tornado_traces.csv'), low_memory=False)
    transaction_df: pd.DataFrame = pd.read_csv(
        os.path.join(args.data_dir, 'tornado_transactions.csv'))
    address_df: pd.DataFrame = pd.read_csv(
        os.path.join(args.data_dir, 'tornadocontracts_abi.csv'),
        names=['address', 'token', 'value', 'name','abi'],
        sep='|')

    deposit_df, withdraw_df = decode_transactions(address_df, transaction_df, trace_df)
    withdraw_df.to_csv(
        os.path.join(args.data_dir, 'complete_withdraw_txs.csv'), index=False)
    deposit_df.to_csv(
        os.path.join(args.data_dir, 'complete_deposit_txs.csv'), index=False)



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to tornado cash deposit data')
    args: Any = parser.parse_args()
    main(args)
