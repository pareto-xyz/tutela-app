"""
Lambda Class's Same Gas Price Heuristic.
"""

import os
from tqdm import tqdm
import pandas as pd
import networkx as nx
from collections import defaultdict
from typing import Any, Tuple, Optional, Dict, List, Set
from src.utils.utils import to_json


def main(args: Any):
    withdraw_df, deposit_df = load_data(args.data_dir)
    clusters, tx2addr = \
        get_same_gas_price_clusters(deposit_df, withdraw_df, by_pool=args.by_pool)
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    appendix: str = '_by_pool' if args.by_pool else ''
    clusters_file: str = os.path.join(
        args.save_dir, f'gas_price_clusters{appendix}.json')
    tx2addr_file: str = os.path.join(
        args.save_dir, f'gas_price_tx2addr{appendix}.json')
    to_json(clusters, clusters_file)
    to_json(tx2addr, tx2addr_file)


def load_data(root) -> Tuple[pd.DataFrame, pd.DataFrame]:
    withdraw_df: pd.DataFrame = pd.read_csv(
        os.path.join(root, 'lighter_complete_withdraw_txs.csv'))

    # Change recipient_address to lowercase.
    withdraw_df['recipient_address'] = withdraw_df['recipient_address'].str.lower()
    
    # Change block_timestamp field to be a timestamp object.
    withdraw_df['block_timestamp'] = withdraw_df['block_timestamp'].apply(pd.Timestamp)

    deposit_df: pd.DataFrame = pd.read_csv(
        os.path.join(root, 'lighter_complete_deposit_txs.csv'))
    
    # Change block_timestamp field to be a timestamp object.
    deposit_df['block_timestamp'] = deposit_df['block_timestamp'].apply(pd.Timestamp)

    return withdraw_df, deposit_df


def get_same_gas_price_clusters(
    deposit_df: pd.DataFrame, 
    withdraw_df: pd.DataFrame, 
    by_pool: bool = False,
) -> Tuple[List[Set[str]], Dict[str, str]]:
    # get deposit transactions with unique gas prices
    unique_gas_deposit_df: pd.DataFrame = filter_by_unique_gas_price(deposit_df)

    # initialize an empty dictionary to store the linked transactions.
    tx2addr: Dict[str, str] = {}
    graph: nx.DiGraph = nx.DiGraph()

    all_withdraws: List[str] = []
    all_deposits: List[str] = []

    # Iterate over the withdraw transactions.
    pbar = tqdm(total=len(withdraw_df))
    for _, withdraw_row in withdraw_df.iterrows():
        
        # apply heuristic for the given withdraw transaction.
        heuristic_fn = same_gas_price_heuristic_by_pool \
            if by_pool else same_gas_price_heuristic
        results: Tuple[bool, pd.Series] = \
            heuristic_fn(withdraw_row, unique_gas_deposit_df)

        # when a deposit transaction matching the withdraw transaction gas price is found, add
        # the linked transactions to the dictionary.
        if results[0]:
            deposit_row: pd.Series = results[1]

            graph.add_node(withdraw_row.hash)
            graph.add_node(deposit_row.hash)
            graph.add_edge(withdraw_row.hash, deposit_row.hash)

            tx2addr[withdraw_row.hash] = withdraw_row.from_address
            tx2addr[deposit_row.hash] = deposit_row.from_address

            all_withdraws.append(withdraw_row.hash)
            all_deposits.append(deposit_row.hash)

        pbar.update()
    pbar.close()

    clusters: List[Set[str]] = [  # ignore singletons
        c for c in nx.weakly_connected_components(graph) if len(c) > 1]

    return clusters, tx2addr


def filter_by_unique_gas_price(transactions_df: pd.DataFrame) -> pd.DataFrame:
    # count the appearances of each gas price in the transactions df
    gas_prices_count = transactions_df['gas_price'].value_counts()

    # filter the gas prices that are unique, i.e., the ones with a count equal to 1
    unique_gas_prices = gas_prices_count[gas_prices_count == 1].keys()

    return transactions_df[transactions_df['gas_price'].isin(unique_gas_prices)]


def same_gas_price_heuristic(
    withdraw_df: pd.DataFrame,
    unique_gas_deposit_df: pd.DataFrame,
) -> Tuple[bool, Optional[pd.Series]]:
    """
    # iterate over each deposit transaction of unique_gas_deposit_df
    for deposit_row in unique_gas_deposit_df.itertuples():
        if ((withdraw_df.gas_price == deposit_row.gas_price) and 
            (withdraw_df.block_timestamp > deposit_row.block_timestamp)):

            return (True, deposit_row)
    """
    searches: pd.DataFrame = unique_gas_deposit_df[
        (unique_gas_deposit_df.gas_price == withdraw_df.gas_price) &
        (unique_gas_deposit_df.block_timestamp < withdraw_df.block_timestamp)
    ]
    if len(searches) > 0:
        return (True, searches.iloc[0])

    return (False, None)


def same_gas_price_heuristic_by_pool(
    withdraw_df: pd.DataFrame, 
    unique_gas_deposit_df: pd.DataFrame,
) -> Tuple[bool, Optional[str]]:
    """
    This heuristic groups together transactions by pool. It is strictly
    a subset of the function `same_gas_price_heuristic`.
    """
    """
    for deposit_row in unique_gas_deposit_df.itertuples():
        # When a deposit transaction with the same gas price as the withdrawal transaction is found, and
        # it also satisfies having an earlier timestamp than it, the tuple (True, deposit_hash) is returned.
        if ((withdraw_df.gas_price == deposit_row.gas_price) and 
            (withdraw_df.block_timestamp > deposit_row.block_timestamp) and 
            (withdraw_df.tornado_cash_address == deposit_row.tornado_cash_address)):

            return (True, deposit_row.hash)
    """
    searches: pd.DataFrame = unique_gas_deposit_df[
        (unique_gas_deposit_df.gas_price == withdraw_df.gas_price) &
        (unique_gas_deposit_df.block_timestamp < withdraw_df.block_timestamp) &
        (unique_gas_deposit_df.tornado_cash_address == withdraw_df.tornado_cash_address)
    ]
    if len(searches) > 0:
        return (True, searches.iloc[0])

    return (False, None)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to tornado cash data')
    parser.add_argument('save_dir', type=str, help='folder to save clusters')
    parser.add_argument('--by-pool', action='store_true', default=False,
                        help='prune by pool heuristic or not?')
    args: Any = parser.parse_args()

    main(args)
