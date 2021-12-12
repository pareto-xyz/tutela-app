"""
Lambda Class's "Same # of Transactions" Heuristic.
"""
import os, json
import itertools
import pandas as pd
from tqdm import tqdm
from typing import Any, Tuple, List, Set, Dict, Optional
from pandas import Timestamp, Timedelta
from src.utils.utils import from_json, to_json, to_pickle
from src.utils.utils import Entity, Heuristic

pd.options.mode.chained_assignment = None

MIN_CONF: float = 0.1
MAX_TIME_DIFF: Timestamp = Timedelta(1, 'hours')


def main(args: Any):
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    appendix: str = '_exact' if args.exact else ''
    clusters_file: str = os.path.join(args.save_dir, f'same_num_txs_clusters{appendix}.json')
    tx2addr_file: str = os.path.join(args.save_dir, f'same_num_txs_tx2addr{appendix}.json')
    addr2conf_file: str = os.path.join(args.save_dir, f'same_num_txs_addr2conf{appendix}.json')
    address_file: str = os.path.join(args.save_dir, f'same_num_txs_address_sets{appendix}.json')
    metadata_file: str = os.path.join(args.save_dir, f'same_num_txs_metadata{appendix}.csv')
   
    withdraw_df, deposit_df, tornado_df = load_data(args.data_dir)
    clusters, address_sets, tx2addr, addr2conf = get_same_num_transactions_clusters(
        deposit_df, withdraw_df, tornado_df, args.data_dir, exact=args.exact)
    
    # save some stuff before continuing
    to_json(clusters, clusters_file)
    to_json(tx2addr, tx2addr_file)
    to_pickle(addr2conf, addr2conf_file)
    del clusters, tx2addr, deposit_df, withdraw_df, tornado_df  # free memory
    to_json(address_sets, address_file)
    """
    address_sets = from_json(address_file)
    addr2conf = from_pickle(addr2conf_file)
    """ 
    address_metadata = get_metadata(address_sets, addr2conf)
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

    withdraw_counts: Dict[str, int] = \
        withdraw_df.recipient_address.value_counts().to_dict()
    deposit_counts: Dict[str, int] = \
        deposit_df.from_address.value_counts().to_dict()

    withdraw_counts: pd.Series = \
        withdraw_df.recipient_address.apply(lambda x: withdraw_counts[x])
    deposit_counts: pd.Series = \
        deposit_df.from_address.apply(lambda x: deposit_counts[x])

    withdraw_df['tx_counts'] = withdraw_counts
    deposit_df['tx_counts'] = deposit_counts

    # Remove withdraw and deposit transactions with only 1 or 2 transactions
    withdraw_df: pd.DataFrame = withdraw_df[withdraw_df.tx_counts > 2]
    deposit_df: pd.DataFrame = deposit_df[deposit_df.tx_counts > 2]

    return withdraw_df, deposit_df, tornado_df


def get_same_num_transactions_clusters(
    deposit_df: pd.DataFrame, 
    withdraw_df: pd.DataFrame, 
    tornado_df: pd.DataFrame,
    data_dir: str,
    exact: bool = False
):
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

    withdraw_hash2time: Dict[str, int] = dict(zip(withdraw_df.hash, withdraw_df.block_timestamp))
    deposit_hash2time: Dict[str, int] = dict(zip(deposit_df.hash, deposit_df.block_timestamp))
    hash2time: Dict[str, int] = {**withdraw_hash2time, **deposit_hash2time}

    tx_clusters: List[Set[str]] = []
    tx2addr: Dict[str, str] = {}
    address_sets: List[Set[str]] = []
    addr2conf: Dict[Tuple[str, str], float] = {}

    print('Processing withdraws')
    pbar = tqdm(total=len(withdraw_df))

    for withdraw_row in withdraw_df.itertuples():
        results = same_num_of_transactions_heuristic(
            withdraw_row, withdraw_df, addr2deposit, tornado_addresses, 
            hash2time, exact = exact)

        if results[0]:
            response_dict = results[1]

            # populate graph with known transactions
            withdraw_txs: List[str] = response_dict['withdraw_txs']
            deposit_txs: List[str] = response_dict['deposit_txs']
            withdraw_tx2addr: Dict[str, str] = response_dict['withdraw_tx2addr']
            deposit_tx2addr: Dict[str, str] = response_dict['deposit_tx2addr']
            tx_cluster: Set[str] = set(withdraw_txs + deposit_txs)

            withdraw_addr: str = response_dict['withdraw_addr']
            deposit_addrs: List[str] = response_dict['deposit_addrs']
            deposit_confs: List[float] = response_dict['deposit_confs']

            for deposit_addr, deposit_conf in zip(deposit_addrs, deposit_confs):
                if withdraw_addr != deposit_addr:
                    address_sets.append([withdraw_addr, deposit_addr])
                    addr2conf[(withdraw_addr, deposit_addr)] = deposit_conf

            tx2addr.update(withdraw_tx2addr)
            tx2addr.update(deposit_tx2addr)
            tx_clusters.append(tx_cluster)

        pbar.update()
    pbar.close()

    print(f'# clusters: {len(tx_clusters)}')

    return tx_clusters, address_sets, tx2addr, addr2conf


def get_metadata(
    address_sets: List[Set[str]],
    addr2conf: Dict[Tuple[str, str], float],
) -> pd.DataFrame:
    """
    Stores metadata about addresses to add to db. 
    """
    address: List[str] = []
    entity: List[int] = [] 
    conf: List[float] = []
    meta_data: List[str] = []
    heuristic: List[int] = []

    pbar = tqdm(total=len(address_sets))
    for cluster in address_sets:
        cluster: List[str] = list(cluster)
        assert len(cluster) == 2
        node_a, node_b = cluster
        conf_ab: float = addr2conf[(node_a, node_b)]
        
        address.append(node_a)
        entity.append(Entity.EOA.value)
        conf.append(conf_ab)
        meta_data.append(json.dumps({}))
        heuristic.append(Heuristic.SAME_NUM_TX.value)

        address.append(node_b)
        entity.append(Entity.EOA.value)
        conf.append(conf_ab)
        meta_data.append(json.dumps({}))
        heuristic.append(Heuristic.SAME_NUM_TX.value)

        pbar.update()
    pbar.close()

    response: Dict[str, List[Any]] = dict(
        address = address,
        entity = entity,
        conf = conf,
        meta_data = meta_data,
        heuristic = heuristic,
    )
    response: pd.DataFrame = pd.DataFrame.from_dict(response)
    response: pd.DataFrame = response.loc[response.groupby('address')['conf'].idxmax()]
    return response


def same_num_of_transactions_heuristic(
    withdraw_tx: pd.Series, 
    withdraw_df: pd.DataFrame, 
    addr2deposit: Dict[str, str], 
    tornado_addresses: Dict[str, int],
    hash2time: Dict[str, int],
    exact: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    # Calculate the number of withdrawals of the address 
    # from the withdraw_tx given as input.
    withdraw_counts, withdraw_set = get_num_of_withdraws(
        withdraw_tx, withdraw_df, tornado_addresses)

    # remove entries that only give to one pool, we are taking 
    # multi-denominational deposits only
    if len(withdraw_counts) == 1:
        return (False, None)

    withdraw_addr: str = withdraw_tx.from_address
    withdraw_txs: List[str] = list(itertools.chain(*list(withdraw_set.values())))
    withdraw_tx2addr = dict(zip(withdraw_txs, 
        [withdraw_addr for _ in range(len(withdraw_txs))]))

    # find block timestamps of all withdraw transactions
    withdraw_times: List[Timestamp] = [hash2time[tx] for tx in withdraw_txs]
    max_withdraw_time_diff: Timestamp = max_time_diff(withdraw_times)

    # if withdraws span too much time, ignore address
    if max_withdraw_time_diff > MAX_TIME_DIFF:
        return (False, None)

    # Based on withdraw_counts, the set of the addresses that have 
    # the same number of deposits is calculated.
    if exact:
        addresses, conf_map = get_same_num_of_deposits(
            withdraw_counts, addr2deposit)
    else:
        addresses, conf_map = get_same_or_more_num_of_deposits(
            withdraw_counts, addr2deposit)

    deposit_addrs: List[str] = list(set(addresses))
    deposit_confs: List[float] = [conf_map[addr] for addr in deposit_addrs]

    deposit_txs: List[str] = []
    deposit_tx2addr: Dict[str, str] = {}

    for address in deposit_addrs:
        deposit_set: Dict[str, List[str]] = addr2deposit[address]
        assert set(withdraw_set.keys()) == set(deposit_set.keys()), \
            "Set of keys do not match."

        address_conf: float = conf_map[address]

        if address_conf >= MIN_CONF:  # must be bigger than 1/2 sure
            # list of all txs for withdraws and deposits regardless of pool
            cur_deposit_txs: List[str] = list(itertools.chain(*list(deposit_set.values())))

            # dictionary from transaction to address
            cur_deposit_tx2addr = dict(zip(cur_deposit_txs, 
                [address for _ in range(len(cur_deposit_txs))]))
            deposit_txs.extend(cur_deposit_txs)
            deposit_tx2addr.update(cur_deposit_tx2addr)

    # find block timestamps of all deposit transactions
    deposit_times: List[Timestamp] = [hash2time[tx] for tx in deposit_txs]
    max_deposit_time_diff: Timestamp = max_time_diff(deposit_times)

    # if deposits span too much time, ignore address
    if max_deposit_time_diff > MAX_TIME_DIFF:
        return (False, None)

    if len(deposit_addrs) > 0:
        privacy_score: float = 1. - 1. / len(deposit_addrs)
        response_dict: Dict[str, Any] = dict(
            withdraw_txs = withdraw_txs,
            deposit_txs = deposit_txs,
            deposit_confs = deposit_confs,
            withdraw_addr = withdraw_addr,
            deposit_addrs = deposit_addrs,
            withdraw_tx2addr = withdraw_tx2addr,
            deposit_tx2addr = deposit_tx2addr,
            privacy_score = privacy_score,
        )
        return (True, response_dict)

    return (False, None)


def get_same_num_of_deposits(
    withdraw_counts: pd.DataFrame, 
    addr2deposit: Dict[str, Dict[str, List[str]]], 
) -> Tuple[List[str], Dict[str, float]]:
    conf_mapping: Dict[str, float] = dict()
    for address, deposits in addr2deposit.items():
        if compare_transactions_exact(withdraw_counts, deposits):
            conf_mapping[address] = 1.0

    addresses: List[str] = list(conf_mapping.keys())
    return addresses, conf_mapping


def get_same_or_more_num_of_deposits(
    withdraw_counts: pd.DataFrame, 
    addr2deposit: Dict[str, Dict[str, List[str]]], 
) -> Tuple[List[str], Dict[str, float]]:
    conf_mapping: Dict[str, float] = dict()
    for address, deposits in addr2deposit.items():
        if compare_transactions(withdraw_counts, deposits):
            num_diff: int = diff_transactions(withdraw_counts, deposits)
            if num_diff == 0:
                conf: float = 1.0
            else:
                conf: float = 1. / num_diff
            conf_mapping[address] = conf

    addresses: List[str] = list(conf_mapping.keys())
    return addresses, conf_mapping


def diff_transactions(
    withdraw_counts_dict: pd.DataFrame, 
    deposit_dict: pd.DataFrame,
) -> int:
    num_diff: int = 0
    for currency in withdraw_counts_dict.keys():
        num_diff += abs(len(deposit_dict[currency]) - withdraw_counts_dict[currency])
    return num_diff


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


def compare_transactions_exact(
    withdraw_counts_dict: pd.DataFrame, 
    deposit_dict: pd.DataFrame,
) -> bool:
    if set(withdraw_counts_dict.keys()) != set(deposit_dict.keys()):
        return False
    for currency in withdraw_counts_dict.keys():
        if not (len(deposit_dict[currency]) == withdraw_counts_dict[currency]):
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
    dictionary with every address to the transactions they
    deposited.

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
    print('building map from address to deposits made by address...')
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


def max_time_diff(times: List[Timestamp]) -> Timestamp:
    diffs: List[Timestamp] = []

    for t1 in times:
        for t2 in times:
            if t1 > t2:
                diffs.append(t1 - t2)
            elif t1 < t2:
                diffs.append(t2 - t1)

    max_diff: Timestamp = max(diffs)
    breakpoint()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to tornado cash data')
    parser.add_argument('tornado_csv', type=str, help='path to tornado cash pool addresses')
    parser.add_argument('save_dir', type=str, help='folder to save matches')
    parser.add_argument('--exact', action='store_true', default=False,
                        help='stricter matching policy')
    args: Any = parser.parse_args()

    main(args)
