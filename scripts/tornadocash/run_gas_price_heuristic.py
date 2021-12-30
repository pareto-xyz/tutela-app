"""
Lambda Class's Same Gas Price Heuristic.
"""

import os, json
from tqdm import tqdm
import pandas as pd
import networkx as nx
from typing import Any, Tuple, Optional, Dict, List, Set, Any
from src.utils.utils import to_json, Entity, Heuristic


def main(args: Any):
    withdraw_df, deposit_df = load_data(args.data_dir)
    clusters, tx2addr = \
        get_same_gas_price_clusters(deposit_df, withdraw_df, by_pool=args.by_pool)
    tx2block, tx2ts = get_transaction_info(withdraw_df, deposit_df)
    address_sets: List[Set[str]] = get_address_sets(clusters, tx2addr)
    address_metadata: List[Dict[str, Any]] = get_metadata(address_sets)
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    appendix: str = '_by_pool' if args.by_pool else ''
    clusters_file: str = os.path.join(
        args.save_dir, f'gas_price_clusters{appendix}.json')
    tx2addr_file: str = os.path.join(
        args.save_dir, f'gas_price_tx2addr{appendix}.json')
    tx2block_file: str = os.path.join(
        args.save_dir, f'gas_price_tx2block{appendix}.json')
    tx2ts_file: str = os.path.join(
        args.save_dir, f'gas_price_tx2ts{appendix}.json')
    address_file: str = os.path.join(
        args.save_dir, f'gas_price_address_set{appendix}.json')
    metadata_file: str = os.path.join(
        args.save_dir, f'gas_price_metadata{appendix}.csv')
    to_json(clusters, clusters_file)
    to_json(tx2addr, tx2addr_file)
    to_json(tx2block, tx2block_file)
    to_json(tx2ts, tx2ts_file)
    to_json(address_sets, address_file)
    address_metadata.to_csv(metadata_file, index=False)


def load_data(root) -> Tuple[pd.DataFrame, pd.DataFrame]:
    withdraw_df: pd.DataFrame = pd.read_csv(
        os.path.join(root, 'lighter_complete_withdraw_txs.csv'))

    # Change recipient_address to lowercase.
    withdraw_df['recipient_address'] = withdraw_df['recipient_address'].str.lower()

    # Change block_timestamp field to be a timestamp object.
    withdraw_df['block_timestamp'] = withdraw_df['block_timestamp'].apply(pd.Timestamp)

    # Remove withdrawals from relayer services. Assume when recipient address is not the 
    # from_address, then this is using a relayer.
    withdraw_df = withdraw_df[withdraw_df['from_address'] == withdraw_df['recipient_address']]

    deposit_df: pd.DataFrame = pd.read_csv(
        os.path.join(root, 'lighter_complete_deposit_txs.csv'))
    
    # Change block_timestamp field to be a timestamp object.
    deposit_df['block_timestamp'] = deposit_df['block_timestamp'].apply(pd.Timestamp)

    return withdraw_df, deposit_df


def get_transaction_info(
    withdraw_df: pd.DataFrame, 
    deposit_df: pd.DataFrame
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    hashes: pd.DataFrame = pd.concat([withdraw_df.hash, deposit_df.hash])
    block_numbers: pd.DataFrame = \
        pd.concat([withdraw_df.block_number, deposit_df.block_number])
    block_timestamps: pd.DataFrame = \
        pd.concat([withdraw_df.block_timestamp, deposit_df.block_timestamp])
    block_timestamps: pd.Series = \
        block_timestamps.apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    tx2block = dict(zip(hashes, block_numbers))
    tx2ts = dict(zip(hashes, block_timestamps))
    return tx2block, tx2ts


def get_same_gas_price_clusters(
    deposit_df: pd.DataFrame, 
    withdraw_df: pd.DataFrame, 
    by_pool: bool = False,
) -> Tuple[List[Set[str]], Dict[str, str]]:
    # get deposit transactions with unique gas prices
    filter_fn = filter_by_unique_gas_price_by_pool if by_pool else filter_by_unique_gas_price
    unique_gas_deposit_df: pd.DataFrame = filter_fn(deposit_df)

    # initialize an empty dictionary to store the linked transactions.
    tx2addr: Dict[str, str] = {}
    graph: nx.DiGraph = nx.DiGraph()
    raw_links: Dict[str, str] = {}  # store non-graph version

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

            raw_links[withdraw_row.hash] = deposit_row.hash

            graph.add_node(withdraw_row.hash)
            graph.add_node(deposit_row.hash)
            graph.add_edge(withdraw_row.hash, deposit_row.hash)

            tx2addr[withdraw_row.hash] = withdraw_row.recipient_address
            tx2addr[deposit_row.hash] = deposit_row.from_address

            all_withdraws.append(withdraw_row.hash)
            all_deposits.append(deposit_row.hash)

        pbar.update()
    pbar.close()

    clusters: List[Set[str]] = [  # ignore singletons
        c for c in nx.weakly_connected_components(graph) if len(c) > 1]

    print(f'# links (graph): {len(clusters)}')
    print(f'# links (raw): {len(raw_links)}')

    return clusters, tx2addr


def get_address_sets(
    tx_clusters: List[Set[str]],
    tx2addr: Dict[str, str],
) -> List[Set[str]]:
    """
    Stores pairs of addresses that are related to each other. Don't 
    apply graphs on this because we are going to join this into the 
    other clusters.
    """
    address_sets: List[Set[str]] = []

    for cluster in tx_clusters:
        addr_set: Set[str] = set([tx2addr[tx] for tx in cluster])
        addr_set: List[str] = list(addr_set)

        if len(addr_set) > 1:  # make sure not singleton
            for addr1 in addr_set:
                for addr2 in addr_set:
                    if addr1 != addr2:
                        address_sets.append({addr1, addr2})

    return address_sets


def get_metadata(address_sets: List[Set[str]]) -> pd.DataFrame:
    """
    Stores metadata about addresses to add to db. 
    """
    unique_addresses: Set[str] = set().union(*address_sets)

    address: List[str] = []
    entity: List[int] = [] 
    conf: List[float] = []
    meta_data: List[str] = []
    heuristic: List[int] = []

    for member in unique_addresses:
        address.append(member)
        entity.append(Entity.EOA.value)
        conf.append(1)
        meta_data.append(json.dumps({}))
        heuristic.append(Heuristic.GAS_PRICE.value)

    response: Dict[str, List[Any]] = dict(
        address = address,
        entity = entity,
        conf = conf,
        meta_data = meta_data,
        heuristic = heuristic,
    )
    response: pd.DataFrame = pd.DataFrame.from_dict(response)
    return response


def filter_by_unique_gas_price(transactions_df: pd.DataFrame) -> pd.DataFrame:
    # count the appearances of each gas price in the transactions df
    gas_prices_count: pd.DataFrame = transactions_df['gas_price'].value_counts()

    # filter the gas prices that are unique, i.e., the ones with a count equal to 1
    unique_gas_prices: pd.DataFrame = gas_prices_count[gas_prices_count == 1].keys()

    return transactions_df[transactions_df['gas_price'].isin(unique_gas_prices)]


def filter_by_unique_gas_price_by_pool(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unlike the non-pool version, we check for unique gas price BY POOL (this
    is a weaker constraint).
    """
    gas_prices_count: pd.DataFrame = transactions_df[['gas_price', 'tornado_cash_address']].value_counts()
    unique_gas_prices: pd.DataFrame = pd.DataFrame(gas_prices_count[gas_prices_count == 1])

    # tuple set with the values (gas_price, tornado_cash_address) is made to filter efficiently
    tuple_set: Set[Any] = set([(row.Index[0], row.Index[1]) for row in unique_gas_prices.itertuples()])

    output_df: pd.DataFrame = pd.DataFrame(
        filter(lambda iter_tuple: \
            (iter_tuple.gas_price, iter_tuple.tornado_cash_address) 
            in tuple_set, transactions_df.itertuples()))

    return output_df


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
