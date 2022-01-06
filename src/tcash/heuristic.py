import os, json
import itertools
import pandas as pd
import networkx as nx
from os.path import join
from datetime import datetime
from collections import defaultdict
from pandas import Timestamp, Timedelta
from typing import Tuple, Dict, List, Set, Any, Optional
from src.utils.utils import Entity, Heuristic


class BaseHeuristic:

    def __init__(self, name: str, tx_root: str, tcash_root: str):
        self._name: str = name
        self._tx_root: str = tx_root
        self._tcash_root: str = tcash_root
        self._out_dir: str = join(tx_root, 'processed')
        os.makedirs(self._out_dir, exist_ok=True)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        withdraw_df: pd.DataFrame = pd.read_csv(
            join(self._tx_root, 'withdraw_txs.csv'))
        # Change recipient_address to lowercase.
        withdraw_df['recipient_address'] = withdraw_df['recipient_address'].str.lower()
        # Change block_timestamp field to be a timestamp object.
        withdraw_df['block_timestamp'] = \
            withdraw_df['block_timestamp'].apply(pd.Timestamp)
        deposit_df: pd.DataFrame = pd.read_csv(
            join(self._tx_root, 'deposit_txs.csv'))
        # Change block_timestamp field to be a timestamp object.
        deposit_df['block_timestamp'] = \
            deposit_df['block_timestamp'].apply(pd.Timestamp)
        # load tornado pool information
        tornado_df: pd.DataFrame = pd.read_csv(
            join(self._tcash_root, 'tornado.csv'))
        return deposit_df, withdraw_df, tornado_df

    def apply_heuristic(
        self,
        deposit_df: pd.DataFrame, 
        withdraw_df: pd.DataFrame,
        tornado_df: pd.DataFrame):
        """
        Expected to return two objects: 

        @clusters: List of Sets of address hashes
        @tx2addrs: Dictionary from transaction to address.
        """
        raise NotImplementedError

    def run(self):
        deposit_df, withdraw_df, tornado_df = self.load_data()
        clusters, tx2addr = self.apply_heuristic(deposit_df, withdraw_df, tornado_df)
        transactions, tx2cluster = get_transactions(clusters)
        tx2block, tx2ts = get_transaction_info(withdraw_df, deposit_df)
        address_sets: List[Set[str]] = get_address_sets(clusters, tx2addr)
        address_table: pd.DataFrame = get_metadata(address_sets)

        transactions: List[str] = list(transactions)
        addresses: List[str] = [tx2addr[tx] for tx in transactions]
        block_numbers: List[int] = [tx2block[tx] for tx in transactions]
        block_timestamps: List[Any] = [tx2ts[tx] for tx in transactions]
        block_timestamps: List[datetime] = [
            datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in block_timestamps]
        clusters: List[int] = [tx2cluster[tx] for tx in transactions]
        meta_datas:  List[str] = [json.dumps({}) for _ in transactions]

        dataset: Dict[str, List[Any]] = {
            'address': addresses,
            'transaction': transactions,
            'block_number': block_numbers,
            'block_ts': block_timestamps,
            'meta_data': meta_datas,
            'cluster': clusters,
        }
        df: pd.DataFrame = pd.DataFrame.from_dict(dataset)
        
        df.to_csv(join(self._out_dir, f'{self._name}.csv'), index=False)
        address_table.to_csv(
            join(self._out_dir, f'{self._name}_address.csv'), index=False)


class ExactMatchHeuristic(BaseHeuristic):

    def __init__(self, name: str, tx_root: str, by_pool: bool = True):
        super().__init__(tx_root)
        self._by_pool: bool = by_pool

    def apply_heuristic(
        self, 
        deposit_df: pd.DataFrame, 
        withdraw_df: pd.DataFrame,
        tornado_df: pd.DataFrame) -> Tuple[List[Set[str]], Dict[str, str]]:
        """
        Iterate over the withdraw transactions and apply heuristic one. For each 
        withdraw with matching deposit transactions, a new element is added to 
        the dictionary with the key as the withdraw hash and the values as all 
        matching deposit hashes.

        It is possible for a transaction hash to appear more than once. As such, 
        we compute weakly connected components to form clusters.
        """
        tx2addr: Dict[str, str] = {}
        graph: nx.DiGraph = nx.DiGraph()
        raw_links: Dict[str, str] = {}

        for withdraw_row in withdraw_df.itertuples():
            results: Tuple[bool, List[pd.Series]] = self.__exact_match_heuristic(
                deposit_df, withdraw_row, by_pool=self.by_pool)

            if results[0]:
                deposit_rows: List[pd.Series] = results[1]
                for deposit_row in deposit_rows:
                    raw_links[withdraw_row.hash] = deposit_row.hash

                    graph.add_node(withdraw_row.hash)
                    graph.add_node(deposit_row.hash)
                    graph.add_edge(withdraw_row.hash, deposit_row.hash)

                    # save transaction -> address map
                    tx2addr[withdraw_row.hash] = withdraw_row.recipient_address
                    tx2addr[deposit_row.hash] = deposit_row.from_address

        clusters: List[Set[str]] = [  # ignore singletons
            c for c in nx.weakly_connected_components(graph) if len(c) > 1]

        return clusters, tx2addr

    def __exact_match_heuristic(
        self,
        deposit_df: pd.DataFrame,
        withdraw_df: pd.DataFrame,
        by_pool: bool = False,
    ) -> Tuple[bool, Optional[List[pd.Series]]]:
        if by_pool:
            matches: pd.DataFrame = deposit_df[
                (deposit_df.from_address == withdraw_df.recipient_address) &
                (deposit_df.block_timestamp < withdraw_df.block_timestamp) & 
                (deposit_df.tornado_cash_address == withdraw_df.tornado_cash_address)
            ]
        else:
            matches: pd.DataFrame = deposit_df[
                (deposit_df.from_address == withdraw_df.recipient_address) &
                (deposit_df.block_timestamp < withdraw_df.block_timestamp)
            ]
        matches: List[pd.Series] = [matches.iloc[i] for i in range(len(matches))]

        if len(matches) > 0:
            return (True, matches)

        return (False, None)


class GasPriceHeuristic(BaseHeuristic):

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Unlike BaseHeuristic.load_data, we ignore relayer transactions.
        """
        withdraw_df: pd.DataFrame = pd.read_csv(
            os.path.join(self._tx_root, 'withdraw_txs.csv'))
        # Change recipient_address to lowercase.
        withdraw_df['recipient_address'] = withdraw_df['recipient_address'].str.lower()
        # Change block_timestamp field to be a timestamp object.
        withdraw_df['block_timestamp'] = \
            withdraw_df['block_timestamp'].apply(pd.Timestamp)
        # Remove withdrawals from relayer services. Assume when recipient address is not the 
        # from_address, then this is using a relayer.
        withdraw_df = withdraw_df[
            withdraw_df['from_address'] == withdraw_df['recipient_address']]
        deposit_df: pd.DataFrame = pd.read_csv(
            os.path.join(self._tx_root, 'deposit_txs.csv'))
        # Change block_timestamp field to be a timestamp object.
        deposit_df['block_timestamp'] = \
            deposit_df['block_timestamp'].apply(pd.Timestamp)
        # load tornado pool information
        tornado_df: pd.DataFrame = pd.read_csv(
            join(self._tcash_root, 'tornado.csv'))
        return deposit_df, withdraw_df, tornado_df

    def apply_heuristic(
        self, 
        deposit_df: pd.DataFrame, 
        withdraw_df: pd.DataFrame,
        tornado_df: pd.DataFrame) -> Tuple[List[Set[str]], Dict[str, str]]:
        """
        Get deposit transactions with unique gas prices.
        """
        filter_fn = self.__filter_by_unique_gas_price_by_pool \
            if self.by_pool else self.__filter_by_unique_gas_price
        unique_gas_deposit_df: pd.DataFrame = filter_fn(deposit_df)

        # initialize an empty dictionary to store the linked transactions.
        tx2addr: Dict[str, str] = {}
        graph: nx.DiGraph = nx.DiGraph()
        raw_links: Dict[str, str] = {}  # store non-graph version

        all_withdraws: List[str] = []
        all_deposits: List[str] = []

        # Iterate over the withdraw transactions.
        for _, withdraw_row in withdraw_df.iterrows():
            # apply heuristic for the given withdraw transaction.
            results: Tuple[bool, pd.Series] = self.__same_gas_price_heuristic(
                unique_gas_deposit_df, withdraw_row, by_pool=self._by_pool)

            # when a deposit transaction matching the withdraw transaction 
            # gas price is found, add the linked transactions to the dictionary.
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

        clusters: List[Set[str]] = [  # ignore singletons
            c for c in nx.weakly_connected_components(graph) if len(c) > 1]

        return clusters, tx2addr

    def __filter_by_unique_gas_price(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        # count the appearances of each gas price in the transactions df
        gas_prices_count: pd.DataFrame = transactions_df['gas_price'].value_counts()
        # filter the gas prices that are unique, i.e., the ones with a count equal to 1
        unique_gas_prices: pd.DataFrame = gas_prices_count[gas_prices_count == 1].keys()

        return transactions_df[transactions_df['gas_price'].isin(unique_gas_prices)]

    def __filter_by_unique_gas_price_by_pool(transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Unlike the non-pool version, we check for unique gas price BY POOL (this
        is a weaker constraint).
        """
        gas_prices_count: pd.DataFrame = transactions_df[
            ['gas_price', 'tornado_cash_address']].value_counts()
        unique_gas_prices: pd.DataFrame = pd.DataFrame(
            gas_prices_count[gas_prices_count == 1])

        # tuple set with the values (gas_price, tornado_cash_address) is made 
        # to filter efficiently
        tuple_set: Set[Any] = set([
            (row.Index[0], row.Index[1]) for row in unique_gas_prices.itertuples()])

        output_df: pd.DataFrame = pd.DataFrame(
            filter(lambda iter_tuple: \
                (iter_tuple.gas_price, iter_tuple.tornado_cash_address) 
                in tuple_set, transactions_df.itertuples()))

        return output_df

    def __same_gas_price_heuristic(
        self,
        deposit_df: pd.DataFrame,
        withdraw_df: pd.DataFrame, 
        by_pool: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        This heuristic groups together transactions by pool. It is strictly
        a subset of the function `same_gas_price_heuristic`.
        """
        if by_pool: 
            searches: pd.DataFrame = deposit_df[
                (deposit_df.gas_price == withdraw_df.gas_price) &
                (deposit_df.block_timestamp < withdraw_df.block_timestamp) &
                (deposit_df.tornado_cash_address == withdraw_df.tornado_cash_address)
            ]
        else:
            searches: pd.DataFrame = deposit_df[
            (deposit_df.gas_price == withdraw_df.gas_price) &
            (deposit_df.block_timestamp < withdraw_df.block_timestamp)
        ]
        if len(searches) > 0:
            return (True, searches.iloc[0])

        return (False, None)


class SameNumTransactionsHeuristic(BaseHeuristic):

    def __init__(self, name: str, tx_root: str, max_num_days: int = 1):
        super().__init__(tx_root)
        self._max_num_days: int = max_num_days

    def apply_heuristic(
        self,
        deposit_df: pd.DataFrame, 
        withdraw_df: pd.DataFrame, 
        tornado_df: pd.DataFrame):
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

        print('Precomputing deposit windows')
        time_window: Timestamp = Timedelta(self._max_num_days, 'days')
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

        raw_portfolios: pd.DataFrame = deposit_windows.apply(
            lambda x: x.iloc[0].groupby('tornado_pool').count()['hash'].to_dict(), axis=1)

        deposit_portfolios: pd.DataFrame = self.__make_portfolio_df(
            raw_portfolios, tornado_tags)

        for withdraw_row in withdraw_df.itertuples():
            results = self.__same_num_of_transactions_heuristic(
                withdraw_row, withdraw_df, deposit_windows, deposit_portfolios, 
                tornado_addresses, self._max_num_days)

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

        return tx_clusters, address_sets, tx2addr, addr2conf

    def __make_portfolio_df(
        raw_portfolios: pd.DataFrame, 
        pools: List[str]) -> pd.DataFrame:
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

    def __same_num_of_transactions_heuristic(
        self,
        withdraw_tx: pd.Series, 
        withdraw_df: pd.DataFrame, 
        deposit_windows: pd.DataFrame,
        deposit_portfolios: pd.DataFrame,
        tornado_addresses: Dict[str, int],
        max_num_days: int,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        # Calculate the number of withdrawals of the address 
        # from the withdraw_tx given as input.
        withdraw_counts, withdraw_set = self.__get_num_of_withdraws(
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

        matched_deposits: List[pd.Dataframe] = self.__get_same_num_of_deposits(
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

    def __get_same_num_of_deposits(
        self,
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

    def __get_num_of_withdraws(
        self,
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

    def run(self):
        deposit_df, withdraw_df, tornado_df = self.load_data()
        clusters, address_sets, tx2addr, addr2conf = \
            self.apply_heuristic(deposit_df, withdraw_df, tornado_df)
        transactions, tx2cluster = get_transactions(clusters)
        tx2block, tx2ts = get_transaction_info(withdraw_df, deposit_df)
        address_sets: List[Set[str]] = get_address_sets(clusters, tx2addr)
        address_table: pd.DataFrame = get_metadata_with_conf(address_sets, addr2conf)

        transactions: List[str] = list(transactions)
        addresses: List[str] = [tx2addr[tx] for tx in transactions]
        block_numbers: List[int] = [tx2block[tx] for tx in transactions]
        block_timestamps: List[Any] = [tx2ts[tx] for tx in transactions]
        block_timestamps: List[datetime] = [
            datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in block_timestamps]
        clusters: List[int] = [tx2cluster[tx] for tx in transactions]
        meta_datas:  List[str] = [json.dumps({}) for _ in transactions]

        dataset: Dict[str, List[Any]] = {
            'address': addresses,
            'transaction': transactions,
            'block_number': block_numbers,
            'block_ts': block_timestamps,
            'meta_data': meta_datas,
            'cluster': clusters,
        }
        df: pd.DataFrame = pd.DataFrame.from_dict(dataset)
        
        df.to_csv(join(self._out_dir, f'{self._name}.csv'), index=False)
        address_table.to_csv(
            join(self._out_dir, f'{self._name}_address.csv'), index=False)


class LinkedTransactionHeuristic(BaseHeuristic):
    pass


class TornMiningHeuristic(BaseHeuristic):
    MINE_POOL_RATES: Dict[str, int] ={
        '0.1 ETH': 4, 
        '1 ETH': 20, 
        '10 ETH': 50, 
        '100 ETH': 400, 
        '100 DAI': 2, 
        '1000 DAI': 10, 
        '10000 DAI': 40, 
        '100000 DAI': 250, 
        '5000 cDAI': 2, 
        '50000 cDAI': 10,
        '500000 cDAI': 40, 
        '5000000 cDAI': 250,
        '0.1 WBTC': 15, 
        '1 WBTC': 120, 
        '10 WBTC': 1000,
    }
    MINE_POOL: List[str] = list(MINE_POOL_RATES.keys())


# -- Helper functions --

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


def get_metadata(
    address_sets: List[Set[str]]) -> pd.DataFrame:
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


def get_metadata_with_conf(
    address_sets: List[Set[str]],
    addr2conf: Dict[Tuple[str, str], float]) -> pd.DataFrame:
    """
    Stores metadata about addresses to add to db. 
    """
    address: List[str] = []
    entity: List[int] = [] 
    conf: List[float] = []
    meta_data: List[str] = []
    heuristic: List[int] = []

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


def get_transactions(clusters: List[Set[str]]) -> Tuple[Set[str], Dict[str, int]]:
    transactions: Set[str] = set()
    tx2cluster: Dict[str, int] = {}
    for c, cluster in enumerate(clusters):
        transactions = transactions.union(cluster)
        for tx in cluster:
            tx2cluster[tx] = c
        
    return transactions, tx2cluster
