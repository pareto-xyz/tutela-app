import os, json, re
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from web3 import Web3
from tqdm import tqdm
from os.path import join
from datetime import datetime
from collections import defaultdict
from pandas import Timestamp, Timedelta
from collections import namedtuple
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

    def load_custom_data(self):
        pass

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
        self.load_custom_data()
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
    """
    If a deposit address matches a withdraw address, then it is trivial to 
    link the two addresses. Therefore, the deposit address needs to be 
    removed from all the other withdraw addresses’ anonymity set. If a 
    number N of deposits with a same address A1, and a number M (M < N) of withdraws 
    with that same address are detected, then a number M-N of deposit transactions
    must be removed from the anonimity set of all the other withdraw transactions.
    """

    def __init__(self, name: str, tx_root: str, tcash_root: str, by_pool: bool = True):
        super().__init__(name, tx_root, tcash_root)
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

        print(f'[{self._name}] looping through rows')
        pbar = tqdm(total=len(withdraw_df))
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

            pbar.update()
        pbar.close()

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
    """
    If there is a deposit and a withdraw transaction with unique gas 
    prices (e.g., 3.1415926 Gwei), then we consider the deposit and 
    the withdraw transactions linked. The corresponding deposit transaction 
    can be removed from any other withdraw transaction’s anonymity set.
    """

    def __init__(self, name: str, tx_root: str, tcash_root: str, by_pool: bool = True):
        super().__init__(name, tx_root, tcash_root)
        self._by_pool: bool = by_pool

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
        print(f'[{self._name}] looping through rows')
        pbar = tqdm(total=len(withdraw_df))
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

            pbar.update()
        pbar.close()

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
    """
    If there are multiple (say 12) deposit transactions coming from a deposit 
    address and later there are 12 withdraw transactions to the same withdraw 
    address, then we can link all these deposit transactions to the withdraw 
    transactions.

    In particular, given a withdrawal transaction, an anonimity score is assigned 
    to it:

    1) The number of previous withdrawal transactions with the same address as the 
    given withdrawal transaction is registered.

    2) The deposit transactions data are grouped by their address. Addresses that 
    deposited the same number of times as the number of withdraws registered, 
    are grouped in a set C.

    3) An anonimity score (of this heuristic) is assigned to the withdrawal 
    transaction following the formula P = 1 - 1/|C|, where P is the anonymity score and  
    is |C| the cardinality of set C.
    """

    def __init__(self, name: str, tx_root: str, tcash_root: str, max_num_days: int = 1):
        super().__init__(name, tx_root, tcash_root)
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

        print(f'[{self._name}] precomputing deposit windows')
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

        print(f'[{self._name}] making portfolio')
        deposit_portfolios: pd.DataFrame = self.__make_portfolio_df(
            raw_portfolios, tornado_tags)

        print(f'[{self._name}] looping through rows')
        pbar = tqdm(total=len(withdraw_df))
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

            pbar.update()
        pbar.close()

        return tx_clusters, address_sets, tx2addr, addr2conf

    def __make_portfolio_df(
        raw_portfolios: pd.DataFrame, 
        pools: List[str]) -> pd.DataFrame:
        raw_portfolios: List[Dict[str, int]] = \
            [eval(x) for x in raw_portfolios['0'].values]
        deposit_portfolios: Dict[str, List[str]] = defaultdict(lambda: [])
        pbar = tqdm(total=len(raw_portfolios))
        for portfolio in raw_portfolios:
            for k in pools:
                if k in portfolio:
                    deposit_portfolios[k].append(portfolio[k])
                else:
                    deposit_portfolios[k].append(0)
            pbar.update()
        pbar.close()
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
        self.load_custom_data()
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
    """
    The main goal of this heuristic is to link Ethereum accounts which interacted 
    with TCash by inspecting Ethereum transactions outside it.

    This is done constructing two sets, one corresponding to the unique TCash 
    deposit addresses and one to the unique TCash withdraw addresses, to 
    then make a query to reveal transactions between addresses of each set.

    When a transaction between two of them is found, TCash deposit transactions 
    done by the deposit address are linked to all the TCash withdraw transactions 
    done by the withdraw address. These two sets of linked transactions are 
    filtered, leaving only the ones that make sense. For example, if a deposit 
    address A is linked to a withdraw address B, but A made a deposit to the 1 
    Eth pool and B made a withdraw to the 10 Eth pool, then this link is not 
    considered. Moreover, when considering a particular link between deposit 
    and withdraw transactions, deposits done posterior to the latest withdraw are 
    removed from the deposit set.
    """
    def __init__(self, name: str, tx_root: str, tcash_root: str, min_interactions: int = 3):
        super().__init__(name, tx_root, tcash_root)
        self._min_interactions: int = min_interactions

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # load tornado pool information
        tornado_df: pd.DataFrame = pd.read_csv(
            join(self._tcash_root, 'tornado.csv'))
        tornado_pools: Dict[str, str] = dict(
            tornado_df.address,
            tornado_df.name.apply(lambda x: x.replace('Tornado Cash Pool', '').strip()),
        )

        withdraw_df: pd.DataFrame = pd.read_csv(
            join(self._tx_root, 'withdraw_txs.csv'))
        withdraw_df['tcash_pool'] = withdraw_df['tornado_cash_address'].apply(
            lambda addr: tornado_pools[addr])
        withdraw_df['block_timestamp'] = \
            withdraw_df['block_timestamp'].apply(pd.Timestamp)
        deposit_df: pd.DataFrame = pd.read_csv(
            join(self._tx_root, 'deposit_txs.csv'))
        deposit_df['tcash_pool'] = deposit_df['tornado_cash_address'].apply(
            lambda addr: tornado_pools[addr])

        return deposit_df, withdraw_df, tornado_df

    def load_custom_data(self):
        print(f'[{self._name}] loading external dataframe')
        external_df: pd.DataFrame = pd.read_csv(
            json(self._tx_root, 'external_txs.csv'))
        external_df: pd.DataFrame = external_df[['from_address', 'to_address']]

        counts: pd.DataFrame = external_df.groupby(['from_address', 'to_address'])\
            .size().reset_index(name='size')
        external_df: pd.DataFrame = counts[counts['size'] >= self._min_interactions]
        external_df: pd.DataFrame = self.__dataframe_from_set_of_sets(
            filter(lambda x: len(x) == 2, 
                self.__filter_repeated_and_permuted(external_df)))

        self.external_df = external_df

    def apply_heuristic(
        self, 
        deposit_df: pd.DataFrame, 
        withdraw_df: pd.DataFrame,
        tornado_df: pd.DataFrame) -> Tuple[List[Set[str]], Dict[str, str]]:

        external_df: pd.DataFrame = self.external_df
        all_tx2addr: Dict[str, str] = {
            **dict(zip(deposit_df.hash, deposit_df.from_address)),
            **dict(zip(withdraw_df.hash, withdraw_df.recipient_address)),
        }
        unique_deposits: Set[str] = set(deposit_df['from_address'])
        unique_withdraws: Set[str] = set(withdraw_df['recipient_address'])

        print(f'[{self._name}] mapping pool to deposit')
        addr_pool_to_deposit: Dict[Tuple[str, str], str] = \
            self.__addresses_and_pools_to_deposits(deposit_df)

        print(f'[{self._name}] mapping withdraw to deposit')
        withdraw2deposit: Dict[str, List[str]] = self.__map_withdraw2deposit(
            external_df, unique_deposits, unique_withdraws)

        print(f'[{self._name}] finding neighbors')
        links: Dict[str, List[str]] = self.__apply_first_neighbors_heuristic(
            withdraw_df, withdraw2deposit, addr_pool_to_deposit)
        
        print(f'[{self._name}] looping through rows')
        clusters, tx2addr = self.__build_clusters(links, all_tx2addr)

        return clusters, tx2addr

    def __apply_first_neighbors_heuristic(
        self,
        withdraw_df: pd.Series,
        withdraw2deposit: Dict[str, str],
        addr_pool_to_deposit: Dict[Tuple[str, str], str]) -> Dict[str, List[str]]:

        links: Dict[str, str] = {}
        pbar = tqdm(len(withdraw_df))
        for row in withdraw_df.itertuples():
            dic = self.__first_neighbors_heuristic(
                row, withdraw2deposit, addr_pool_to_deposit)
            links.update(dic)
            pbar.update()
        pbar.close()

        return dict(filter(lambda elem: len(elem[1]) != 0, links.items()))

    def __first_neighbors_heuristic(
        withdraw_df: pd.Series,
        withdraw2deposit: Dict[str, str],
        addr_pool_to_deposit: Dict[Tuple[str, str], str]) -> Dict[str, List[str]]:
        """
        Check that there has been a transaction between this address and some deposit
        address outside Tcash. If not, return an empty list for this particular withdraw.
        """
        address: str = withdraw_df.recipient_address
        pool: str = withdraw_df.tcash_pool

        AddressPool: namedtuple = namedtuple('AddressPool', ['address', 'pool'])

        if address in withdraw2deposit.keys():
            interacted_addresses: List[str] = withdraw2deposit[address]
            linked_deposits: List[str] = []

            for addr in interacted_addresses:
                if AddressPool(address=addr, pool=pool) in addr_pool_to_deposit.keys():
                    for d in addr_pool_to_deposit[AddressPool(address=addr, pool=pool)]:
                        if d.timestamp < withdraw_df.block_timestamp:
                            linked_deposits.append(d.deposit_hash)
                            
            return {withdraw_df.hash: linked_deposits}
        else:
            return {withdraw_df.hash: []}

    def __build_clusters(
        self, 
        links: Dict[str, List[str]],
        all_tx2addr: Dict[str, str]) -> Tuple[List[Set[str]], Dict[str, str]]:

        graph: nx.DiGraph = nx.DiGraph()
        tx2addr: Dict[str, str] = {}

        pbar = tqdm(total=len(links))
        for withdraw, deposits in links.items():
            graph.add_node(withdraw)
            graph.add_nodes_from(deposits)

            for deposit in deposits:
                graph.add_edge(withdraw, deposit)

                tx2addr[withdraw] = all_tx2addr[withdraw]
                tx2addr[deposit] = all_tx2addr[deposit]

            pbar.update()
        pbar.close()

        clusters: List[Set[str]] = [  # ignore singletons
            c for c in nx.weakly_connected_components(graph) if len(c) > 1]

        return clusters, tx2addr

    def __map_withdraw2deposit(
        self,
        address_and_withdraw: pd.DataFrame,
        deposits: Set[str],
        withdraws: Set[str]
    ) -> Dict[str, List[str]]:
        """
        Map interactions between every withdraw address to every deposit address, outside TCash
        """
        deposit_and_withdraw: np.array = np.empty((0, 2), dtype=str)
        pbar = tqdm(total=len(address_and_withdraw))
        for row in address_and_withdraw.itertuples():

            if (self.__is_D_W_tx(row.address_1, row.address_2, deposits, withdraws) or 
                self.__is_D_DW_tx(row.address_1, row.address_2, deposits, withdraws) or 
                self.__is_DW_W_tx(row.address_1, row.address_2, deposits, withdraws)):
                deposit_and_withdraw: np.array = np.append(
                    deposit_and_withdraw, [[row.address_1, row.address_2]], axis=0)

            elif (self.__is_W_D_tx(row.address_1, row.address_2, deposits, withdraws) or 
                    self.__is_W_DW_tx(row.address_1, row.address_2, deposits, withdraws) or 
                    self.__is_DW_D_tx(row.address_1, row.address_2, deposits, withdraws)):
                deposit_and_withdraw: np.array = np.append(
                    deposit_and_withdraw, [[row.address_2, row.address_1]], axis=0)
                
            elif self.__is_DW_DW_tx(row.address_1, row.address_2, deposits, withdraws):
                deposit_and_withdraw: np.array = np.append(
                    deposit_and_withdraw, [[row.address_1, row.address_2]], axis=0)
                deposit_and_withdraw: np.array = np.append(
                    deposit_and_withdraw, [[row.address_2, row.address_1]], axis=0)
            else:
                raise ValueError('Unknown type: D_W, W_D, D_DW, DW_D, W_DW, DW_W, DW_DW')

            pbar.update()
        pbar.close()

        D_W_df: pd.DataFrame = pd.DataFrame(
            deposit_and_withdraw, columns=['deposit', 'withdraw'])

        return dict(D_W_df.groupby('withdraw')['deposit'].apply(list))

    def __addresses_and_pools_to_deposits(
        self, deposit_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Gives a dictionary with deposit addresses as keys and the 
        deposit transactions each address made as values.
        """
        addresses_and_pools: dict = dict(
            deposit_df.groupby('from_address')['tcash_pool'].apply(list))
        addresses_and_pools_to_deposits: dict = {}
        pbar = tqdm(total=len(addresses_and_pools))
        for addr in addresses_and_pools.keys():
            for pool in addresses_and_pools[addr]:
                addresses_and_pools_to_deposits.update(
                    self.__addr_pool_to_deposits(addr, pool, deposit_df))
            pbar.update()
        pbar.close()

        return addresses_and_pools_to_deposits

    def __addr_pool_to_deposits(
        self, address: str, tcash_pool: str, deposit_df) -> dict:
        """
        Given an address and the TCash pool, give all the deposits that 
        address has done in that pool.
        """
        mask = (deposit_df['from_address'] == address) & \
            (deposit_df['tcash_pool'] == tcash_pool)

        addr_pool_deposits: pd.DataFrame = deposit_df[mask]
        HashTimestamp = namedtuple('HashTimestamp', ['deposit_hash', 'timestamp'])
        AddressPool = namedtuple('AddressPool', ['address', 'pool'])

        hashes_and_timestamps: List[Optional[HashTimestamp]] = \
            [None] * len(addr_pool_deposits)
        for i, row in enumerate(addr_pool_deposits.itertuples()):
            hashes_and_timestamps[i] = HashTimestamp(
                deposit_hash=row.hash, timestamp=row.block_timestamp)

        return {AddressPool(address=address, pool=tcash_pool): hashes_and_timestamps}

    def __filter_repeated_and_permuted(
        self, external_df: pd.DataFrame) -> Set[Set[str]]:
        filtered_set = set()
        for row in external_df.itertuples():
            filtered_set.add(frozenset([row.from_address, row.to_address]))
        return filtered_set

    def __dataframe_from_set_of_sets(
        self, set_of_sets: Set[Set[str]]) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame({'address_1':[], 'address_2':[]})

        for s in set_of_sets:
            s_tuple: Tuple[str] = tuple(s)
            if len(s) == 2:
                df: pd.DataFrame = df.append(
                    {'address_1': s_tuple[0], 'address_2': s_tuple[1]}, 
                    ignore_index=True)
            else:
                df: pd.DataFrame = df.append(
                    {'address_1': s_tuple[0], 'address_2': s_tuple[0]}, 
                    ignore_index=True)

        return df

    def __is_D_type(
        self, address: str, deposits: Set[str], withdraws: Set[str]) -> bool:
        return (address in deposits) and (address not in withdraws)

    def __is_W_type(
        self, address: str, deposits: Set[str], withdraws: Set[str]) -> bool:
        return (address not in deposits) and (address in withdraws)

    def __is_DW_type(
        self, address: str, deposits: Set[str], withdraws: Set[str]) -> bool:
        return (address in deposits) and (address in withdraws)

    def __is_D_W_tx(
        self, address1: str, address2: str, 
        deposits: Set[str], withdraws: Set[str]) -> bool:
        return self.__is_D_type(address1, deposits, withdraws) and \
            self.__is_W_type(address2, deposits, withdraws)

    def __is_W_D_tx(
        self, address1: str, address2: str,
        deposits: Set[str], withdraws: Set[str]) -> bool:
        return self.__is_W_type(address1, deposits, withdraws) and \
            self.__is_D_type(address2, deposits, withdraws)

    def is_D_DW_tx(
        self, address1: str, address2: str,
        deposits: Set[str], withdraws: Set[str]) -> bool:
        return self.__is_D_type(address1, deposits, withdraws) and \
            self.__is_DW_type(address2, deposits, withdraws)

    def __is_DW_D_tx(
        self, address1: str, address2: str,
        deposits: Set[str], withdraws: Set[str]) -> bool:
        return self.__is_DW_type(address1, deposits, withdraws) and \
            self.__is_D_type(address2, deposits, withdraws)

    def __is_W_DW_tx(
        self, address1: str, address2: str,
        deposits: Set[str], withdraws: Set[str]) -> bool:
        return self.__is_W_type(address1, deposits, withdraws) and \
            self.__is_DW_type(address2, deposits, withdraws)

    def __is_DW_W_tx(
        self, address1: str, address2: str,
        deposits: Set[str], withdraws: Set[str]) -> bool:
        return self.__is_DW_type(address1, deposits, withdraws) and \
            self.__is_W_type(address2, deposits, withdraws)

    def __is_DW_DW_tx(
        self, address1: str, address2: str,
        deposits: Set[str], withdraws: Set[str]) -> bool:
        return self.__is_DW_type(address1, deposits, withdraws) and \
            self.__is_DW_type(address2, deposits, withdraws)


class TornMiningHeuristic(BaseHeuristic):
    """
    Through Anonimity Mining, TCash users are able to receive TORN token as a 
    reward for using the application. This is done in a sequence of two steps:

    Anonimity Points (AP) are claimed for already spent notes. The quantity of 
    AP obtained depends on how many blocks the note was in a TCash pool, and 
    the pool where it was.
    
    Using the TCash Automated Market Maker (AMM), users can exchange anonymity 
    points for TORN. From a pure data point of view, this actions are seen as 
    transactions with the TCash Miner. An example of transaction (1) can be seen 
    here and an example of transaction (2) can be seen here.

    What can be seen clearly by decoding the input data field of these transactions 
    is that transactions of type (1) call the TCash miner method reward and the 
    ones of type (2) the method withdraw. Transactions of type (2) are the ones we 
    are interested in, since they give us the following information:

    - Amount of AP being swapped to TORN.
    - Address of TORN recipient.

    Making the assumption that users are swapping the totality of their AP, and 
    that those AP have been claimed for a single note, then we can link deposit 
    and withdrawal transactions with the following procedure:

    1. Check if the recipient address of a transaction of type (2) has done a 
    deposit or a withdraw in TCash.

    2. If the address is included in the set of deposit (withdraw) addreses, then 
    check all deposits (withdraws) of that address, convert AP into number of 
    blocks according to the pool the deposit was done, and then search for a 
    withdraw transaction with block number equal to

        deposit_blocks + AP_converted_blocks = withdraw_blocks

    3. When such a withdraw (deposit) is found, then the two transactions are 
    considered linked.

    Results are given in a dictionary, each element has the following structure:

        (withdraw_hash, withdraw_address, AP_converted_blocks): 
        [(deposit_hash, deposit_address), ...]

    In this compact manner we have the information about the withdraw transactions 
    with their addresses and their linking to deposit transactions with their 
    addresses. We also have the amount of blocks that the notes were deposited in 
    the TCash pool.
    """
    MINE_POOL_RATES: Dict[str, int] = {
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

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # load tornado pool information
        tornado_df: pd.DataFrame = pd.read_csv(
            join(self._tcash_root, 'tornado.csv'))
        tornado_pools: Dict[str, str] = dict(
            tornado_df.address,
            tornado_df.name.apply(lambda x: x.replace('Tornado Cash Pool', '').strip()),
        )

        withdraw_df: pd.DataFrame = pd.read_csv(
            join(self._tx_root, 'withdraw_txs.csv'))
        withdraw_df['tcash_pool'] = withdraw_df['tornado_cash_address'].apply(
            lambda addr: tornado_pools[addr])
        withdraw_df['block_timestamp'] = \
            withdraw_df['block_timestamp'].apply(pd.Timestamp)
        deposit_df: pd.DataFrame = pd.read_csv(
            join(self._tx_root, 'deposit_txs.csv'))
        deposit_df['tcash_pool'] = deposit_df['tornado_cash_address'].apply(
            lambda addr: tornado_pools[addr])

        return deposit_df, withdraw_df, tornado_df

    def load_custom_data(self):
        print(f'[{self._name}] loading miner dataframe')
        miner_df: pd.DataFrame = pd.read_csv(
            join(self._tx_root, 'miner_txs.csv'))
        true = True; false = False

        miner_abi_df: pd.DataFrame = pd.read_csv(
            join(self._tcash_root, 'miner_abi.csv'), 
            names=['address', 'abi'],
            sep='|')
        miner_address: str = miner_abi_df.address.iloc[0]
        miner_abi: str = miner_abi_df.abi.iloc[0]

        w3: Web3 = Web3(Web3.HTTPProvider("https://cloudflare-eth.com"))
        contract: w3.eth.contract = w3.eth.contract(
            address=w3.toChecksumAddress(miner_address), 
            abi=eval(miner_abi))

        def decode_miner_txs(tx, contract):
            """
            Decodes input field of a miner tx.

            Returns a tuple of three elements:
            1. Function call
            2. Anonimity points
            3. Recipient address

            When function call is not of type 'withdraw', 2. and 3. are NaN.
            """
            func_object, func_params = contract.decode_function_input(tx.input)
            f = str(func_object)
            
            fn_call = 'withdraw' if re.search('withdraw', f) else 'reward' \
                if re.search('reward', f) else 'other'

            if fn_call in ['reward', 'other']:
                return (fn_call, np.nan, np.nan)
            else:
                anonimity_points = func_params['_args'][0]
                recipient_address = func_params['_args'][2][1]
                return (fn_call, anonimity_points, recipient_address.lower())

        decoded_info: pd.DataFrame = miner_df.apply(
            lambda row: decode_miner_txs(row, contract), axis=1)

        # Get three lists from the decoded_info results
        fn_calls = np.empty(len(decoded_info), dtype=str)
        anonimity_points = np.empty(len(decoded_info), dtype=object)
        recipient_addresses = np.empty(len(decoded_info), dtype=object)

        for i in range(len(decoded_info)):
            fn_calls[i] = decoded_info[i][0]
            anonimity_points[i] = decoded_info[i][1]
            recipient_addresses[i] = decoded_info[i][2]

        miner_df['function_call'] = fn_calls
        miner_df['anonimity_points'] = anonimity_points
        miner_df['recipient_address'] = recipient_addresses

        # drop input field so that data is lighter, now that we have 
        # extracted the necessary information
        miner_df = miner_df.drop(columns=['input'])

        self.miner_df = miner_df

    def apply_heuristic(
        self, 
        deposit_df: pd.DataFrame, 
        withdraw_df: pd.DataFrame,
        tornado_df: pd.DataFrame) -> Tuple[List[Set[str]], Dict[str, str]]:

        miner_df = self.miner_df
        miner_df = miner_df[miner_df['function_call'] == 'w']
        miner_df.head()

        mining_pools: List[str] = list(self.MINE_POOL_RATES.keys())

        # drop all the txs that are not from the mining pools
        deposit_df: pd.DataFrame = \
            deposit_df[deposit_df['tcash_pool'].isin(mining_pools)]
        withdraw_df: pd.DataFrame = \
            withdraw_df[withdraw_df['tcash_pool'].isin(mining_pools)]

        unique_deposits = set(deposit_df['from_address'])
        unique_withdraws = set(withdraw_df['recipient_address'])

        addr2deposits: Dict[str, Any] = \
            self.__address_to_txs_and_blocks(deposit_df, 'deposit')
        addr2withdraws: Dict[str, Any] = \
            self.__address_to_txs_and_blocks(withdraw_df, 'withdraw')

        print(f'[{self._name}] computing links')
        total_linked_txs: Dict[str, Dict[str, Any]] = \
            self.__get_total_linked_txs(
                miner_df, unique_deposits, unique_withdraws, 
                addr2deposits, addr2withdraws)

        print(f'[{self._name}] mapping withdraw to deposit')
        w2d: Dict[Tuple[str], List[Tuple[str]]] = \
            self.__apply_anonymity_mining_heuristic(total_linked_txs)

        print(f'[{self._name}] looping through rows')
        clusters, tx2addr = self.__build_clusters(w2d)

        return clusters, tx2addr

    def __address_to_txs_and_blocks(
        self, txs_df: pd.DataFrame, tx_type: str) -> Dict[str, Any]:
        assert tx_type in ['deposit', 'withdraw'], 'Transaction type error'
        address_field: str = 'from_address' if tx_type == 'deposit' else 'recipient_address'
        addr_to_txs_and_blocks: Dict[str, Any] = {}

        for _, row in txs_df.iterrows():
            if row[address_field] not in addr_to_txs_and_blocks.keys():
                addr_to_txs_and_blocks[row[address_field]] = \
                    {row.tcash_pool: [(row.hash, row.block_number)]}
            elif row.tcash_pool not in addr_to_txs_and_blocks[row[address_field]].keys():
                addr_to_txs_and_blocks[row[address_field]].update(
                    {row.tcash_pool: [(row.hash, row.block_number)]})
            else:
                addr_to_txs_and_blocks[row[address_field]][row.tcash_pool].append(
                    (row.hash, row.block_number))

        return addr_to_txs_and_blocks

    def __apply_anonymity_mining_heuristic(
        self,
        total_linked_txs: Dict[str, Dict[str, Any]],
    ) -> Dict[Tuple[str], List[Tuple[str]]]:
        """
        The final version of the results is obtained applying this function
        to the output of the 'apply_anonimity_mining_heuristic' function.

        w2d -> withdraws and blocks to deposits
        """
        w2d: Dict[Tuple[str], List[Tuple[str]]] = {}

        pbar = tqdm(total=len(total_linked_txs['W']))
        for addr in total_linked_txs['W'].keys():
            for hsh in total_linked_txs['W'][addr]:
                delta_blocks: float = total_linked_txs['W'][addr][hsh][0][2]
                w2d[(hsh, addr, delta_blocks)] = [
                    (t[0],t[1]) for t in total_linked_txs['W'][addr][hsh]]
            pbar.update()
        pbar.close()

        pbar = tqdm(total=len(total_linked_txs['D']))
        for addr in total_linked_txs['D'].keys():
            for hsh in total_linked_txs['D'][addr]:
                for tx_tuple in total_linked_txs['D'][addr][hsh]:
                    if tx_tuple[0] not in w2d.keys():
                        w2d[tuple(tx_tuple)] = [(hsh, addr)]
                    else:
                        if (hsh, addr) not in w2d[tx_tuple]:
                            w2d[tuple(tx_tuple)].append((hsh, addr))
            pbar.update()
        pbar.close()

        return w2d

    def __build_clusters(links: Any) -> Tuple[List[Set[str]], Dict[str, str]]:
        graph: nx.DiGraph = nx.DiGraph()
        tx2addr: Dict[str, str] = {}

        pbar = tqdm(total=len(links))
        for withdraw_tuple, deposit_tuples in links.items():
            withdraw_tx, withdraw_addr, _ = withdraw_tuple
            graph.add_node(withdraw_tx)
            tx2addr[withdraw_tx] = withdraw_addr

            for deposit_tuple in deposit_tuples:
                deposit_tx, deposit_addr = deposit_tuple
                graph.add_node(deposit_tx)
                graph.add_edge(withdraw_tx, deposit_tx)
                tx2addr[deposit_tx] = deposit_addr

            pbar.update()
        pbar.close()

        clusters: List[Set[str]] = [  # ignore singletons
            c for c in nx.weakly_connected_components(graph) if len(c) > 1]

        return clusters, tx2addr

    def __get_total_linked_txs(
        self,
        miner_txs: pd.Series,
        unique_deposits: Set[str],
        unique_withdraws: Set[str],
        addr2deposits: Dict[str, Any], 
        addr2withdraws: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        total_linked_txs: Dict[str, Dict[str, Any]] = {'D': {}, 'W': {}}

        pbar = tqdm(total=len(miner_txs))
        for miner_tx in miner_txs.itertuples():
            linked_txs: Dict[str, Dict[str, Any]] = \
                self.__anonymity_mining_heuristic(
                    miner_tx, unique_deposits, unique_withdraws, 
                    addr2deposits, addr2withdraws)
            if len(linked_txs) != 0:
                if 'D' in linked_txs.keys():
                    if len(linked_txs['D']) != 0:
                        total_linked_txs['D'].update(linked_txs['D'])
                if 'W' in linked_txs.keys():
                    if len(linked_txs['W']) != 0:
                        total_linked_txs['W'].update(linked_txs['W'])
            pbar.update()
        pbar.close()

        return total_linked_txs

    def __anonymity_mining_heuristic(
        self,
        miner_tx: pd.Series,
        unique_deposits: Set[str],
        unique_withdraws: Set[str],
        addr2deposits: Dict[str, Any],
        addr2withdraws: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        linked_txs: Dict[str, Dict[str, Any]] = {}

        if self.__is_D_type(
            miner_tx.recipient_address, unique_deposits, unique_withdraws):
            d_dict: Dict[str, Any] = self.__D_type_anonymity_heuristic(
                miner_tx, addr2deposits, addr2withdraws)
            if len(d_dict[miner_tx.recipient_address]) != 0:
                linked_txs['D'] = d_dict
            return linked_txs
        elif self.__is_W_type(
            miner_tx.recipient_address, unique_deposits, unique_withdraws):
            w_dict: Dict[str, Any] = self.__W_type_anonymity_heuristic(
                miner_tx, addr2deposits, addr2withdraws)
            if len(w_dict[miner_tx.recipient_address]) != 0:
                linked_txs['W'] = w_dict
            return linked_txs
        elif self.__is_DW_type(
            miner_tx.recipient_address, unique_deposits, unique_withdraws):
            d_dict: Dict[str, Any] = self.__D_type_anonymity_heuristic(
                miner_tx, addr2deposits, addr2withdraws)
            if len(d_dict[miner_tx.recipient_address]) != 0:
                linked_txs['D'] = d_dict
            w_dict: Dict[str, Any] = self.__W_type_anonymity_heuristic(
                miner_tx, addr2deposits, addr2withdraws)
            if len(w_dict[miner_tx.recipient_address]) != 0:
                linked_txs['W'] = w_dict
            return linked_txs

        return linked_txs

    def __is_D_type(self, address: str, deposits: Set[str], withdraws: Set[str]):
        return (address in deposits) and (address not in withdraws)

    def __is_W_type(self, address: str, deposits: Set[str], withdraws: Set[str]):
        return (address not in deposits) and (address in withdraws)

    def __is_DW_type(self, address: str, deposits: Set[str], withdraws: Set[str]):
        return (address in deposits) and (address in withdraws)

    def __ap2blocks(self, anonymity_points: int, pool: str) -> float:
        rate = self.MINE_POOL_RATES[pool]
        return anonymity_points / float(rate)

    def __D_type_anonymity_heuristic(
        self,
        miner_tx: pd.Series, 
        addr2deposits: Dict[str, Any],
        addr2withdraws: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        d_addr: str = miner_tx.recipient_address
        d_addr2w: Dict[str, Dict[str, Any]] = {d_addr: {}}

        for d_pool in addr2deposits[d_addr]:
            for (d_hash, d_blocks) in addr2deposits[d_addr][d_pool]:
                delta_blocks: float = self.__ap2blocks(miner_tx.anonimity_points, d_pool)

                for w_addr in addr2withdraws.keys():
                    if d_pool in addr2withdraws[w_addr].keys():
                        for (w_hash, w_blocks) in addr2withdraws[w_addr][d_pool]:
                            if d_blocks + delta_blocks == w_blocks:
                                if d_hash not in d_addr2w[d_addr].keys():
                                    d_addr2w[d_addr][d_hash] = [(w_hash, w_addr, delta_blocks)]
                                else:
                                    d_addr2w[d_addr][d_hash].append((w_hash, w_addr, delta_blocks))

        return d_addr2w

    def __W_type_anonymity_heuristic(
        self,
        miner_tx: pd.Series, 
        addr2deposits: Dict[str, Any],
        addr2withdraws: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        w_addr: str = miner_tx.recipient_address
        w_addr2d: Dict[str, Dict[str, Any]] = {w_addr: {}}
        
        for w_pool in addr2withdraws[w_addr]:
            for (w_hash, w_blocks) in addr2withdraws[w_addr][w_pool]:
                delta_blocks: float = self.__ap2blocks(miner_tx.anonimity_points, w_pool)

                for d_addr in addr2deposits.keys():
                    if w_pool in addr2deposits[d_addr].keys():
                        for (d_hash, d_blocks) in addr2deposits[d_addr][w_pool]:
                            if d_blocks + delta_blocks == w_blocks:
                                if w_hash not in w_addr2d[w_addr].keys():
                                    w_addr2d[w_addr][w_hash] = [(d_hash, d_addr, delta_blocks)]
                                else:
                                    w_addr2d[w_addr][w_hash].append((d_hash, d_addr, delta_blocks))

        return w_addr2d

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
