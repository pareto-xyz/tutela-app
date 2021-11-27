"""
Lambda Class's "Same # of Transactions" Heuristic.
"""
import os, json
import itertools
import pandas as pd
from tqdm import tqdm
import networkx as nx
from typing import Any, Tuple, List, Set, Dict, Optional
from src.utils.utils import from_json, to_json, Entity, Heuristic

pd.options.mode.chained_assignment = None


def main(args: Any):
    withdraw_df, deposit_df, tornado_df = load_data(args.data_dir)
    clusters, address_sets, tx2addr = get_same_num_transactions_clusters(
        deposit_df, withdraw_df, tornado_df, args.data_dir)
    address_metadata = get_metadata(address_sets)
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    clusters_file: str = os.path.join(args.save_dir, f'same_num_txs_clusters.json')
    tx2addr_file: str = os.path.join(args.save_dir, f'same_num_txs_tx2addr.json')
    address_file: str = os.path.join(args.save_dir, f'same_num_txs_address_sets.json')
    metadata_file: str = os.path.join(args.save_dir, f'same_num_txs_metadata.csv')
    to_json(clusters, clusters_file)
    to_json(tx2addr, tx2addr_file)
    to_json(address_sets, address_file)
    address_metadata.to_csv(metadata_file, index=False)


def load_data(root) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    # Load Tornado data
    tornado_df: pd.DataFrame = pd.read_csv(args.tornado_csv)

    return withdraw_df, deposit_df, tornado_df


def get_same_num_transactions_clusters(
    deposit_df: pd.DataFrame, 
    withdraw_df: pd.DataFrame, 
    tornado_df: pd.DataFrame,
    data_dir: str,
) -> Tuple[List[Set[str]], List[Set[str]], Dict[str, str]]:
    """
    Same Number of Transactions Heuristic.

    If there are multiple (say 12) deposit transactions coming from 
    a deposit address and later there are 12 withdraw transactions 
    to the same withdraw address, *then we can link all these deposit 
    transactions to the withdraw transactions*. 
    """
    tornado_addresses: Dict[str, int] = \
        dict(zip(tornado_df.address, tornado_df.tags))

    cached_addr2deposit: str =  os.path.join(data_dir, 'same_num_txs_addr2deposit.json')
    if os.path.isfile(cached_addr2deposit):
        print('Found cached deposit mapping: loading...')
        addr2deposit: Dict[str, Dict[str, List[str]]] = from_json(cached_addr2deposit)
    else:
        addr2deposit = get_address_deposits(deposit_df, tornado_addresses)
        to_json(addr2deposit, cached_addr2deposit)

    # initialize an empty dictionary
    tx2addr: Dict[str, str] = {}
    tx_graph: nx.DiGraph = nx.DiGraph()
    address_sets: List[Set[str]] = []

    print('Processing withdraws')
    pbar = tqdm(total=len(withdraw_df))
    for withdraw_row in withdraw_df.itertuples():
        results = same_num_of_transactions_heuristic(
            withdraw_row, withdraw_df, addr2deposit, tornado_addresses)

        if results[0]:
            response_dict = results[1]

            # populate graph with known transactions
            withdraw_txs: List[str] = response_dict['withdraw_txs']
            deposit_txs: List[str] = response_dict['deposit_txs']
            withdraw_tx2addr: Dict[str, str] = response_dict['withdraw_tx2addr']
            deposit_tx2addr: Dict[str, str] = response_dict['deposit_tx2addr']

            tx_graph.add_nodes_from(withdraw_txs)
            tx_graph.add_nodes_from(deposit_txs)
            edge_txs: List[Tuple(str, str)] = \
                list(itertools.product(withdraw_txs, deposit_txs))
            tx_graph.add_edges_from(edge_txs)

            # store related addresses
            withdraw_addr: str = response_dict['withdraw_addr']
            deposit_addrs: Set[str] = response_dict['deposit_addrs']
            address_sets.extend([
                {withdraw_addr, deposit_addr} for deposit_addr in deposit_addrs])

            # upload to tx2addr
            tx2addr.update(withdraw_tx2addr)
            tx2addr.update(deposit_tx2addr)

        pbar.update()
    pbar.close()

    tx_clusters: List[Set[str]] = [
        c for c in nx.weakly_connected_components(tx_graph) if len(c) > 1]

    return tx_clusters, address_sets, tx2addr


def same_num_of_transactions_heuristic(
    withdraw_tx: pd.Series, 
    withdraw_df: pd.DataFrame, 
    addr2deposit: Dict[str, str], 
    tornado_addresses: Dict[str, int],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    # Calculate the number of withdrawals of the address 
    # from the withdraw_tx given as input.
    withdraw_counts, withdraw_set = get_num_of_withdraws(
        withdraw_tx, withdraw_df, tornado_addresses)

    withdraw_addr: str = withdraw_tx.from_address
    withdraw_txs: List[str] = list(itertools.chain(*list(withdraw_set.values())))
    withdraw_tx2addr = dict(zip(withdraw_txs, 
        [withdraw_addr for _ in range(len(withdraw_txs))]))

    # Based on withdraw_counts, the set of the addresses that have 
    # the same number of deposits is calculated.
    addresses: List[str] = get_same_or_more_num_of_deposits(
        withdraw_counts, addr2deposit)
    deposit_addrs: Set[str] = set(addresses)

    deposit_txs: List[str] = list()
    deposit_tx2addr: Dict[str, str] = {}

    for address in addresses:
        deposit_set: Dict[str, List[str]] = addr2deposit[address]
        assert set(withdraw_set.keys()) == set(deposit_set.keys()), \
            "Set of keys do not match."

        # list of all txs for withdraws and deposits regardless of pool
        cur_deposit_txs: List[str] = list(itertools.chain(*list(deposit_set.values())))

        # dictionary from transaction to address
        cur_deposit_tx2addr = dict(zip(cur_deposit_txs, 
            [address for _ in range(len(cur_deposit_txs))]))
        deposit_txs.extend(cur_deposit_txs)
        deposit_tx2addr.update(cur_deposit_tx2addr)

    if len(addresses) > 0:
        response_dict: Dict[str, Any] = dict(
            withdraw_txs = withdraw_txs,
            deposit_txs = deposit_txs,
            withdraw_addr = withdraw_addr,
            deposit_addrs = deposit_addrs,
            withdraw_tx2addr = withdraw_tx2addr,
            deposit_tx2addr = deposit_tx2addr,
        )
        return (True, response_dict)

    return (False, None)


def get_same_or_more_num_of_deposits(
    withdraw_counts: pd.DataFrame, 
    addr2deposit: Dict[str, Dict[str, List[str]]], 
) -> List[str]:
    result: Dict[str, Any] = dict(
        filter( lambda elem: compare_transactions(withdraw_counts, elem[1]), 
                addr2deposit.items()))
    return list(result.keys())


def compare_transactions(
    withdraw_counts_dict: pd.DataFrame, 
    deposit_dict: pd.DataFrame,
) -> bool:
    """
    Given two dictionaries, withdraw_dict and deposit_dict representing 
    the total deposits and withdraws made by an address to each TCash pool, 
    respectively, compares if the set of keys of both are equal and when 
    they are, checks if all values in the deposit dictionary are equal or 
    greater than each of the corresponding values of the withdraw 
    dicionary. If this is the case, returns True, if not, False.
    """
    if set(withdraw_counts_dict.keys()) != set(deposit_dict.keys()):
        return False
    for currency in withdraw_counts_dict.keys():
        if not (len(deposit_dict[currency]) >= withdraw_counts_dict[currency]):
            return False
    return True


def get_num_of_withdraws(
    withdraw_tx: pd.Series, 
    withdraw_df: pd.DataFrame, 
    tornado_addresses: Dict[str, str],
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Given a particular withdraw transaction and the withdraw transactions 
    DataFrame, gets the total withdraws the address made in each pool. It 
    is returned as a dictionary with the pools as the keys and the number 
    of withdraws as the values.
    """
    cur_withdraw_pool: str = tornado_addresses[withdraw_tx.tornado_cash_address]

    withdraw_txs: Dict[str, List[str]] = {
        tornado_addresses[withdraw_tx.tornado_cash_address]: []}

    subset_df: pd.DataFrame = withdraw_df[
        (withdraw_df.recipient_address == withdraw_tx.recipient_address) & 
        (withdraw_df.block_timestamp <= withdraw_tx.block_timestamp) & 
        (withdraw_df.hash != withdraw_tx.hash)
    ]
    subset_df['tornado_pool'] = subset_df.tornado_cash_address.map(
        lambda x: tornado_addresses[x])

    withdraw_count: pd.DataFrame = subset_df.groupby('tornado_pool').size()
    withdraw_count: Dict[str, int] = withdraw_count.to_dict()

    withdraw_txs: pd.DataFrame = subset_df.groupby('tornado_pool')['hash'].apply(list)
    withdraw_txs: Dict[str, List[str]] = withdraw_txs.to_dict()

    # add 1 for current address
    if cur_withdraw_pool in withdraw_count:
        withdraw_count[cur_withdraw_pool] += 1
        withdraw_txs[cur_withdraw_pool].append(withdraw_tx.hash)
    else:
        withdraw_count[cur_withdraw_pool] = 1
        withdraw_txs[cur_withdraw_pool] = [withdraw_tx.hash]

    return withdraw_count, withdraw_txs


def get_address_deposits(
    deposit_df: pd.DataFrame,
    tornado_addresses: Dict[str, int],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Given the deposit transactions DataFrame, returns a 
    dictionary with every address of the deposit

    Example:
    {
        '0x16e54b35d789832440ab47ae765e6a8098280676': 
            {
                '0.1 ETH': [...],
                '100 USDT': [...],
            },
        '0x35dd029618f4e1835008da21fd98850c776453f0': {
            '0.1 ETH': [...],
        },
        '0xe906442c11b85acbc58eccb253b9a55a20b80a56': {
            '0.1 ETH': [...],
        },
        '0xaf301de836c81deb8dff9dc22745e23c476155b2': {
            '1 ETH': [...],
            '0.1 ETH': [...],
            '10 ETH': [...],
        },
    }
    """
    counts_df: pd.DataFrame = pd.DataFrame(
        deposit_df[['from_address', 'tornado_cash_address']].value_counts()
    ).rename(columns={0: "count"})
    
    addr2deposit: Dict[str, str] = {}
    print('building map from address to deposits...')
    pbar = tqdm(total=len(counts_df))
    for row in counts_df.itertuples():
        deposit_set: pd.Series = deposit_df[
            (deposit_df.from_address == row.Index[0]) &
            (deposit_df.tornado_cash_address == row.Index[1])
        ].hash
        deposit_set: Set[str] = set(deposit_set)

        if row.Index[0] in addr2deposit.keys():
            addr2deposit[row.Index[0]][
                tornado_addresses[row.Index[1]]] = deposit_set
        else:
            addr2deposit[row.Index[0]] = {
                tornado_addresses[row.Index[1]]: deposit_set}

        pbar.update()
    pbar.close()

    return addr2deposit


def get_metadata(address_sets: List[Set[str]]) -> pd.DataFrame:
    """
    Stores metadata about addresses to add to db. 
    """
    address: List[str] = []
    entity: List[int] = [] 
    conf: List[float] = []
    meta_data: List[str] = []
    cluster_type: List[int] = []

    pbar = tqdm(total=len(address_sets))
    for cluster in address_sets:
        for member in cluster:
            address.append(member)
            entity.append(Entity.EOA.value)
            conf.append(1)
            meta_data.append(json.dumps({}))
            cluster_type.append(Heuristic.SAME_NUM_TX.value)
        pbar.update()
    pbar.close()

    response: Dict[str, List[Any]] = dict(
        address = address,
        entity = entity,
        conf = conf,
        meta_data = meta_data,
        cluster_type = cluster_type,
    )
    response: pd.DataFrame = pd.DataFrame.from_dict(response)
    return response


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to tornado cash data')
    parser.add_argument('tornado_csv', type=str, help='path to tornado cash pool addresses')
    parser.add_argument('save_dir', type=str, help='folder to save matches')
    args: Any = parser.parse_args()

    main(args)
