"""
Lambda Class's Linked Transactions Heuristic.
"""

import os, json
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from collections import namedtuple
from typing import Any, Tuple, Dict, Set, Any, List
from src.utils.utils import to_json, Entity, Heuristic


def main(args: Any):
    root: str = args.data_dir

    with open(os.path.join(root, 'tornado_pools.json')) as json_file:
        tornado_addresses: Dict[str, str] = json.load(json_file)

    deposit_txs: pd.DataFrame = pd.read_csv(
        os.path.join(root, 'lighter_complete_deposit_txs.csv'))
    deposit_txs['tcash_pool'] = deposit_txs['tornado_cash_address']\
        .apply(lambda addr: tornado_addresses[addr])
    withdraw_txs: pd.DataFrame = pd.read_csv(
        os.path.join(root, 'lighter_complete_withdraw_txs.csv'))
    withdraw_txs['tcash_pool'] = withdraw_txs['tornado_cash_address']\
        .apply(lambda addr: tornado_addresses[addr])

    all_tx2addr: Dict[str, str] = {
        **dict(deposit_txs.hash, deposit_txs.from_address),
        **dict(withdraw_txs.hash, withdraw_txs.recipient_address),
    }

    addr_pool_to_deposit: Dict[Tuple[str, str], str] = \
        load_addresses_and_pools_to_deposits_json(
            os.path.join(root, 'addresses_and_pools_to_deposits.json'))

    address_and_withdraw: pd.DataFrame = pd.read_csv(
        os.path.join(root, 'transactions_between_deposit_and_withdraw_addresses.csv'))
    address_and_withdraw: pd.DataFrame = \
        address_and_withdraw[['from_address', 'to_address']]

    address_and_withdraw: pd.DataFrame = dataframe_from_set_of_sets(
        filter(lambda x: len(x) == 2, 
            filter_repeated_and_permuted(address_and_withdraw)))

    unique_deposits: Set[str] = set(deposit_txs['from_address'])
    unique_withdraws: Set[str] = set(withdraw_txs['recipient_address'])

    withdraw2deposit: Dict[str, str] = map_withdraw2deposit(
        address_and_withdraw, unique_deposits, unique_withdraws)

    links: Dict[str, List[str]] = apply_first_neighbors_heuristic(
        withdraw_txs, withdraw2deposit, addr_pool_to_deposit)

    # build a graph, then find clusters, build tx2addr
    clusters, tx2addr = build_clusters(links, all_tx2addr)

    address_sets: List[Set[str]] = get_address_sets(clusters, tx2addr)
    address_metadata: List[Dict[str, Any]] = get_metadata(address_sets)
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)

    clusters_file: str = os.path.join(args.save_dir, f'linked_tx_clusters.json')
    tx2addr_file: str = os.path.join(args.save_dir, f'linked_tx_tx2addr.json')
    address_file: str = os.path.join(args.save_dir, f'linked_tx_address_set.json')
    metadata_file: str = os.path.join(args.save_dir, f'linked_tx_metadata.csv')
    to_json(clusters, clusters_file)
    to_json(tx2addr, tx2addr_file)
    to_json(address_sets, address_file)
    address_metadata.to_csv(metadata_file, index=False)


def build_clusters(
    links: Dict[str, List[str]],
    all_tx2addr: Dict[str, str]) -> Tuple[List[Set[str]], Dict[str, str]]:

    graph: nx.DiGraph = nx.DiGraph()
    tx2addr: Dict[str, str] = {}

    for withdraw, deposits in links:
        graph.add_node(withdraw)
        graph.add_nodes_from(deposits)

        for deposit in deposits:
            graph.add_edge(withdraw, deposit)

            tx2addr[withdraw] = all_tx2addr[withdraw]
            tx2addr[deposit] = all_tx2addr[deposit]

    clusters: List[Set[str]] = [  # ignore singletons
        c for c in nx.weakly_connected_components(graph) if len(c) > 1]

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
        heuristic.append(Heuristic.LINKED_TX.value)

    response: Dict[str, List[Any]] = dict(
        address = address,
        entity = entity,
        conf = conf,
        meta_data = meta_data,
        heuristic = heuristic,
    )
    response: pd.DataFrame = pd.DataFrame.from_dict(response)
    return response


def apply_first_neighbors_heuristic(
    withdraw_txs: pd.Series,
    withdraw2deposit: Dict[str, str],
    addr_pool_to_deposit: Dict[Tuple[str, str], str]) -> Dict[str, List[str]]:

    links: Dict[str, str] = {}
    for row in tqdm(withdraw_txs.itertuples(), total=len(withdraw_txs)):
        dic = first_neighbors_heuristic(row, withdraw2deposit, addr_pool_to_deposit)
        links.update(dic)

    return dict(filter(lambda elem: len(elem[1]) != 0, links.items()))


def first_neighbors_heuristic(
    withdraw_tx: pd.Series,
    withdraw2deposit: Dict[str, str],
    addr_pool_to_deposit: Dict[Tuple[str, str], str]) -> dict:
    """
    Check that there has been a transaction between this address and some deposit
    address outside Tcash. If not, return an empty list for this particular withdraw.
    """
    address: str = withdraw_tx.recipient_address
    pool: str = withdraw_tx.tcash_pool

    AddressPool = namedtuple('AddressPool', ['address', 'pool'])

    if address in withdraw2deposit.keys():
        interacted_addresses = withdraw2deposit[address]
        linked_deposits = []

        for addr in interacted_addresses:
            if AddressPool(address=addr, pool=pool) in addr_pool_to_deposit.keys():
                for d in addr_pool_to_deposit[AddressPool(address=addr, pool=pool)]:
                    if d.timestamp < withdraw_tx.block_timestamp:
                        linked_deposits.append(d.deposit_hash)
                        
        return {withdraw_tx.hash: linked_deposits}
    else:
        return {withdraw_tx.hash: []}


def map_withdraw2deposit(
    address_and_withdraw: pd.DataFrame,
    deposits: Set[str],
    withdraws: Set[str]
) -> dict:
    """
     Map interactions between every withdraw address to every deposit address, outside TCash
    """
    deposit_and_withdraw = np.empty((0, 2), dtype=str)
    
    for row in tqdm(address_and_withdraw.itertuples(), 
                    total=len(address_and_withdraw), 
                    mininterval=0.5):

        if (is_D_W_tx(row.address_1, row.address_2, deposits, withdraws) or 
            is_D_DW_tx(row.address_1, row.address_2, deposits, withdraws) or 
            is_DW_W_tx(row.address_1, row.address_2, deposits, withdraws)):
            deposit_and_withdraw = np.append(
                deposit_and_withdraw, [[row.address_1, row.address_2]], axis=0)

        elif (is_W_D_tx(row.address_1, row.address_2, deposits, withdraws) or 
                is_W_DW_tx(row.address_1, row.address_2, deposits, withdraws) or 
                is_DW_D_tx(row.address_1, row.address_2, deposits, withdraws)):
            deposit_and_withdraw = np.append(
                deposit_and_withdraw, [[row.address_2, row.address_1]], axis=0)
            
        elif is_DW_DW_tx(row.address_1, row.address_2, deposits, withdraws):
            deposit_and_withdraw = np.append(
                deposit_and_withdraw, [[row.address_1, row.address_2]], axis=0)
            deposit_and_withdraw = np.append(
                deposit_and_withdraw, [[row.address_2, row.address_1]], axis=0)
        else:
            raise ValueError('Unknown type: D_W, W_D, D_DW, DW_D, W_DW, DW_W, DW_DW')

    D_W_df = pd.DataFrame(deposit_and_withdraw, columns=['deposit', 'withdraw'])
    return dict(D_W_df.groupby('withdraw')['deposit'].apply(list))

# -- tx classification utilities --

def is_D_type(address: str, deposits: Set[str], withdraws: Set[str]) -> bool:
    return (address in deposits) and (address not in withdraws)


def is_W_type(address: str, deposits: Set[str], withdraws: Set[str]) -> bool:
    return (address not in deposits) and (address in withdraws)


def is_DW_type(address: str, deposits: Set[str], withdraws: Set[str]) -> bool:
    return (address in deposits) and (address in withdraws)


def is_D_W_tx(
    address1: str, address2: str, 
    deposits: Set[str], withdraws: Set[str]) -> bool:
    return is_D_type(address1, deposits, withdraws) and \
        is_W_type(address2, deposits, withdraws)


def is_W_D_tx(
    address1: str, address2: str,
    deposits: Set[str], withdraws: Set[str]) -> bool:
    return is_W_type(address1, deposits, withdraws) and \
        is_D_type(address2, deposits, withdraws)


def is_D_DW_tx(
    address1: str, address2: str,
    deposits: Set[str], withdraws: Set[str]) -> bool:
    return is_D_type(address1, deposits, withdraws) and \
        is_DW_type(address2, deposits, withdraws)


def is_DW_D_tx(
    address1: str, address2: str,
    deposits: Set[str], withdraws: Set[str]) -> bool:
    return is_DW_type(address1, deposits, withdraws) and \
         is_D_type(address2, deposits, withdraws)


def is_W_DW_tx(
    address1: str, address2: str,
    deposits: Set[str], withdraws: Set[str]) -> bool:
    return is_W_type(address1, deposits, withdraws) and \
         is_DW_type(address2, deposits, withdraws)


def is_DW_W_tx(
    address1: str, address2: str,
    deposits: Set[str], withdraws: Set[str]) -> bool:
    return is_DW_type(address1, deposits, withdraws) and \
         is_W_type(address2, deposits, withdraws)


def is_DW_DW_tx(
    address1: str, address2: str,
    deposits: Set[str], withdraws: Set[str]) -> bool:
    return is_DW_type(address1, deposits, withdraws) and \
         is_DW_type(address2, deposits, withdraws)

# -- data utilities -- 

def filter_repeated_and_permuted(address_and_withdraw_df):
    filtered_addresses_set = set()

    for row in address_and_withdraw_df.itertuples():
        filtered_addresses_set.add(frozenset([row.from_address, row.to_address]))
    
    return filtered_addresses_set


def dataframe_from_set_of_sets(set_of_sets):
    addresses_df = pd.DataFrame({'address_1':[], 'address_2':[]})
        
    for s in tqdm(set_of_sets):
        s_tuple = tuple(s)
        if len(s) == 2:
            addresses_df = addresses_df.append(
                {'address_1': s_tuple[0], 'address_2': s_tuple[1]}, 
                ignore_index=True)
        else:
            addresses_df = addresses_df.append(
                {'address_1': s_tuple[0], 'address_2': s_tuple[0]}, 
                ignore_index=True)

    return addresses_df


def remap_keys(mapping):
    return [{'key': k,'value': v} for k, v in mapping.items()]


def load_addresses_and_pools_to_deposits_json(filepath):
    with open(filepath) as json_file:
        raw_dict_list = json.load(json_file)
        addresses_and_pools_to_deposits: dict = {}

        HashTimestamp = namedtuple('HashTimestamp', ['deposit_hash', 'timestamp'])
        AddressPool = namedtuple('AddressPool', ['address', 'pool'])

        for dic in raw_dict_list:
            elem = {
                AddressPool(address=dic['key'][0], pool=dic['key'][1]): \
                    [HashTimestamp(deposit_hash=l[0], timestamp=l[1]) for l in dic['value']]
            }
            addresses_and_pools_to_deposits.update(elem)
        
        return addresses_and_pools_to_deposits


def _addr_pool_to_deposits(address: str, tcash_pool: str, deposit_txs) -> dict:
    """
    # Given an address and the TCash pool, give all the deposits that 
    # address has done in that pool.
    """
    mask = (deposit_txs['from_address'] == address) & \
        (deposit_txs['tcash_pool'] == tcash_pool)

    addr_pool_deposits = deposit_txs[mask]
    
    HashTimestamp = namedtuple('HashTimestamp', ['deposit_hash', 'timestamp'])
    AddressPool = namedtuple('AddressPool', ['address', 'pool'])
    
    hashes_and_timestamps: list = [None] * len(addr_pool_deposits)
    for i, row in enumerate(addr_pool_deposits.itertuples()):
        hashes_and_timestamps[i] = HashTimestamp(
            deposit_hash=row.hash, timestamp=row.block_timestamp)

    return {AddressPool(address=address, pool=tcash_pool): hashes_and_timestamps}


def addresses_and_pools_to_deposits(deposit_txs) -> dict:
    """
    Gives a dictionary with deposit addresses as keys and the 
    deposit transactions each address made as values.
    """
    addresses_and_pools: dict = dict(
        deposit_txs.groupby('from_address')['tcash_pool'].apply(list))
    addresses_and_pools_to_deposits: dict = {}    
    for addr in tqdm(addresses_and_pools.keys(), mininterval=3):
        for pool in addresses_and_pools[addr]:
            addresses_and_pools_to_deposits.update(
                _addr_pool_to_deposits(addr, pool, deposit_txs))

    return addresses_and_pools_to_deposits


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to tornado cash data')
    parser.add_argument('save_dir', type=str, help='folder to save clusters')
    args: Any = parser.parse_args()

    main(args)

