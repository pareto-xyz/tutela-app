"""
Lambda Class's "Same # of Transactions" Heuristic.
"""
import os, json
import itertools
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import Any, Tuple, List, Set, Dict, Optional
from pandas import Timestamp, Timedelta
from src.utils.utils import to_json, from_pickle, to_pickle
from src.utils.utils import Entity, Heuristic

pd.options.mode.chained_assignment = None


def main(args: Any):
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    appendix: str = f'_exact_{args.max_num_days}days'
    clusters_file: str = os.path.join(args.save_dir, f'same_num_txs_clusters{appendix}.json')
    tx2addr_file: str = os.path.join(args.save_dir, f'same_num_txs_tx2addr{appendix}.json')
    tx2block_file: str = os.path.join(args.save_dir, f'same_num_txs_tx2block{appendix}.json')
    tx2ts_file: str = os.path.join(args.save_dir, f'same_num_txs_tx2ts{appendix}.json')
    addr2conf_file: str = os.path.join(args.save_dir, f'same_num_txs_addr2conf{appendix}.json')
    address_file: str = os.path.join(args.save_dir, f'same_num_txs_address_sets{appendix}.json')
    metadata_file: str = os.path.join(args.save_dir, f'same_num_txs_metadata{appendix}.csv')
   
    withdraw_df, deposit_df, tornado_df = load_data(args.data_dir)
    clusters, address_sets, tx2addr, addr2conf = get_same_num_transactions_clusters(
        deposit_df, withdraw_df, tornado_df, args.max_num_days, args.data_dir)
    tx2block, tx2ts = get_transaction_info(withdraw_df, deposit_df)

    # save some stuff before continuing
    to_json(clusters, clusters_file)
    to_json(tx2addr, tx2addr_file)
    to_json(tx2block, tx2block_file)
    to_json(tx2ts, tx2ts_file)
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

    return withdraw_df, deposit_df, tornado_df


def get_transaction_info(
    withdraw_df: pd.DataFrame, 
    deposit_df: pd.DataFrame
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    hashes: pd.DataFrame = pd.concat([withdraw_df.hash, deposit_df.hash])
    block_numbers: pd.DataFrame = \
        pd.concat([withdraw_df.block_number, deposit_df.block_number])
    block_timestamps: pd.DataFrame = \
        pd.concat([withdraw_df.block_timestamp, deposit_df.block_timestamp])
    block_timestamps: pd.DataFrame = block_timestamps.apply(pd.Timestamp)
    block_timestamps: pd.Series = \
        block_timestamps.apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    tx2block = dict(zip(hashes, block_numbers))
    tx2ts = dict(zip(hashes, block_timestamps))
    return tx2block, tx2ts


def get_same_num_transactions_clusters(
    deposit_df: pd.DataFrame, 
    withdraw_df: pd.DataFrame, 
    tornado_df: pd.DataFrame,
    max_num_days: int,
    data_dir: str,
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
    tornado_tags: List[str] = tornado_df.tags.to_list()

    tx_clusters: List[Set[str]] = []
    tx2addr: Dict[str, str] = {}
    address_sets: List[Set[str]] = []
    addr2conf: Dict[Tuple[str, str], float] = {}

    cache_window_file: str = os.path.join(
        data_dir, f'deposit_windows_{max_num_days}days.pickle')
    cache_portfolio_file: str = os.path.join(
        data_dir, f'deposit_portfolio_{max_num_days}days.csv')

    if os.path.isfile(cache_window_file):
        print('Loading deposit windows')
        deposit_windows: pd.DataFrame = from_pickle(cache_window_file)
        raw_portfolios: pd.DataFrame = pd.read_csv(cache_portfolio_file)
    else:
        print('Precomputing deposit windows')
        time_window: Timestamp = Timedelta(max_num_days, 'days')
        deposit_df['tornado_pool'] = deposit_df.tornado_cash_address.map(
            lambda x: tornado_addresses[x])
        deposit_windows: pd.Series = deposit_df.apply(lambda x: deposit_df[
            # find all deposits made before current one
            (deposit_df.block_timestamp <= x.block_timestamp) & 
            # find all deposits made at most 24 hr before current one
            (deposit_df.block_timestamp >= (x.block_timestamp - time_window)) &
            # only consider those with same address as current one
            (deposit_df.from_address == x.from_address) &
            # ignore the current one from returned set
            (deposit_df.hash != x.hash)
        ], axis=1)
        deposit_windows: pd.DataFrame = pd.DataFrame(deposit_windows)
        to_pickle(deposit_windows, cache_window_file)

        raw_portfolios: pd.DataFrame = deposit_windows.apply(
            lambda x: x.iloc[0].groupby('tornado_pool').count()['hash'].to_dict(), axis=1)
        raw_portfolios.to_csv(cache_portfolio_file, index=False)

    deposit_portfolios: pd.DataFrame = make_portfolio_df(raw_portfolios, tornado_tags)

    print('Processing withdraws')
    pbar = tqdm(total=len(withdraw_df))

    for withdraw_row in withdraw_df.itertuples():
        results = same_num_of_transactions_heuristic(
            withdraw_row, withdraw_df, deposit_windows, deposit_portfolios, 
            tornado_addresses, max_num_days)

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
    deposit_windows: pd.DataFrame,
    deposit_portfolios: pd.DataFrame,
    tornado_addresses: Dict[str, int],
    max_num_days: int,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    # Calculate the number of withdrawals of the address 
    # from the withdraw_tx given as input.
    withdraw_counts, withdraw_set = get_num_of_withdraws(
        withdraw_tx, withdraw_df, tornado_addresses, max_num_days = max_num_days)

    # remove entries that only give to one pool, we are taking 
    # multi-denominational deposits only
    if len(withdraw_counts) == 1:
        return (False, None)

    # if there are only 1 or 2 txs, ignore
    if sum(withdraw_counts.values()) < 2:
        return (False, None)

    withdraw_addr: str = withdraw_tx.recipient_address  # who's gets the withdrawn
    withdraw_txs: List[str] = list(itertools.chain(*list(withdraw_set.values())))
    withdraw_tx2addr = dict(zip(withdraw_txs, 
        [withdraw_addr for _ in range(len(withdraw_txs))]))

    matched_deposits: List[pd.Dataframe] = get_same_num_of_deposits(
        withdraw_counts, deposit_windows, deposit_portfolios)

    if len(matched_deposits) == 0:  # no matched deposits by heuristic
        return (False, None)

    deposit_addrs: List[str] = []
    deposit_txs: List[str] = []
    deposit_confs: List[float] = []
    deposit_tx2addr: Dict[str, str] = {}

    for match in matched_deposits:
        deposit_addrs.append(match.from_address.iloc[0])
        txs: List[str] = match.hash.to_list()
        deposit_txs.extend(txs)
        deposit_confs.extend([1.0] * len(txs))
        deposit_tx2addr.update(dict(zip(match.hash, match.from_address)))

    deposit_addrs: List[str] = list(set(deposit_addrs))

    privacy_score: float = 1. - 1. / len(matched_deposits)
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


def get_same_num_of_deposits(
    withdraw_counts: pd.DataFrame, 
    deposit_windows: pd.DataFrame,
    deposit_portfolios: pd.DataFrame,
) -> List[pd.DataFrame]:
    # simple assertion that the number of non-zero currencies must be the same
    mask: Optional[pd.DataFrame] = \
        (deposit_portfolios > 0).sum(axis=1) == len(withdraw_counts)
    for k, v in withdraw_counts.items():
        if mask is None:
            mask: pd.DataFrame = (deposit_portfolios[k] == v)
        else:
            mask: pd.DataFrame = mask & (deposit_portfolios[k] == v)
    return [x[0] for x in deposit_windows[mask].values]


def make_portfolio_df(raw_portfolios: pd.DataFrame, pools: List[str]) -> pd.DataFrame:
    raw_portfolios: List[Dict[str, int]] = \
        [eval(x) for x in raw_portfolios['0'].values]
    deposit_portfolios: Dict[str, List[str]] = defaultdict(lambda: [])
    for portfolio in raw_portfolios:
        for k in pools:
            if k in portfolio:
                deposit_portfolios[k].append(portfolio[k])
            else:
                deposit_portfolios[k].append(0)
    deposit_portfolios: Dict[str, List[str]] = dict(deposit_portfolios)
    return pd.DataFrame.from_dict(deposit_portfolios)


def make_deposit_df(
    deposits: Dict[str, List[str]],
    hash2time: Dict[str, Timestamp],
) -> pd.DataFrame:
    transactions: List[str] = []
    pools: List[str] = []
    timestamps: List[Timestamp] = []
    for pool, txs in deposits.items():
        transactions.extend(txs)
        pools.extend([pool] * len(txs))
        timestamps.extend([hash2time[tx] for tx in txs])
    out: Dict[str, Any] = {
        'transaction': transactions,
        'pool': pools,
        'timestamp': timestamps,
    }
    out: pd.DataFrame = pd.DataFrame.from_dict(out)
    return out


def make_address_deposit_df(
    addr2deposit: Dict[str, Dict[str, List[str]]],
    hash2time: Dict[str, Timestamp],
) -> pd.DataFrame:
    addr_deposit_df: List[pd.DataFrame] = []
    for address, deposits in addr2deposit.items():
        deposit_df: pd.DataFrame = make_deposit_df(deposits, hash2time)
        deposit_df['address'] = address
        addr_deposit_df.append(deposit_df)
    addr_deposit_df: pd.DataFrame = pd.concat(addr_deposit_df)
    addr_deposit_df: pd.DataFrame = addr_deposit_df.reset_index()
    return addr_deposit_df


def get_num_of_withdraws(
    withdraw_tx: pd.Series, 
    withdraw_df: pd.DataFrame, 
    tornado_addresses: Dict[str, str],
    max_num_days: int,
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

    time_window: Timestamp = Timedelta(max_num_days, 'days')

    subset_df: pd.DataFrame = withdraw_df[
        # ignore txs made by others
        (withdraw_df.recipient_address == withdraw_tx.recipient_address) & 
        # ignore future transactions
        (withdraw_df.block_timestamp <= withdraw_tx.block_timestamp) &
        # ignore other withdraw transactions not within the last MAX_TIME_DIFF  
        (withdraw_df.block_timestamp >= (withdraw_tx.block_timestamp - time_window)) &
        # ignore the query row
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


def get_max_time_diff(times: List[Timestamp]) -> float:
    diffs: List[float] = []

    for t1, t2 in itertools.product(times, repeat=2):
        diffs.append(abs((t1 - t2).total_seconds()))

    return max(diffs)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to tornado cash data')
    parser.add_argument('tornado_csv', type=str, help='path to tornado cash pool addresses')
    parser.add_argument('save_dir', type=str, help='folder to save matches')
    parser.add_argument('--max-num-days', type=int, default=1, 
                        help='number of maximum days (default: 1)')
    args: Any = parser.parse_args()

    main(args)
