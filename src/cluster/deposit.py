import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from src.utils.utils import Entity
from src.utils.loader import DataframeLoader
from src.cluster.base import BaseCluster


class DepositCluster(BaseCluster):
    """
    Cluster addresses by deposit address reuse heuristic.

    See Algorithm 1: https://fc20.ifca.ai/preproceedings/31.pdf
    """

    def __init__(
        self,
        loader: DataframeLoader,
        a_max: float = 0.01,  # max amount diff (in ether)
        t_max: float = 3200,  # max time diff (in blocks)
        save_dir: str = './',
    ):
        super().__init__(loader)

        self.a_max: float = a_max
        self.t_max: float = t_max
        self.save_dir: str = save_dir

    def make_clusters(self):
        last_chunk: pd.DataFrame = pd.DataFrame() 
        
        # assumes a maximum of 10k txs per block. 
        max_txs_per_block: int = 10000
        chunk_size: int = max_txs_per_block * self.t_max
        chunk_count: int = 0

        # save data about (unique) addresses
        metadata_file: str = os.path.join(self.save_dir, 'metadata.csv')
        result_file: str = os.path.join(self.save_dir, 'data.csv')

        print('processing txs',  end = '', flush=True)

        for tx_chunk in self.loader.yield_transactions(chunk_size):
            # make numeric and convert wei -> eth 
            tx_chunk.value = tx_chunk.value.astype(float) / 10**18
            min_block: int = tx_chunk.block_number.min()
            max_block: int = tx_chunk.block_number.max()

            # join the last chunk and current chunk
            both_chunks: pd.DataFrame = last_chunk.append(tx_chunk)

            result: pd.DataFrame = self._cluster_chunk(
                both_chunks,
                self.loader.get_exchanges(),
                self.loader.get_miners(),
                self.loader.get_blacklist(),
            )

            """
            Add confidence to dataframe.

            NOTE: some clusters may contain multiple deposits. For now, 
            we opt for the simple thing and allow multiple deposits in 
            the same cluster to have different confidences. All EOAs 
            attached to each deposit will have its corresponding `conf`. 

            An alternative strategy may be to set all elements in the 
            cluster to a minimum value, although both strategies have 
            pros and cons. For example, if one deposit has conf 0.99 and
            the other conf 0.01, it seems wrong to set the former to 0.01.
            """
            scores: pd.DataFrame = self._get_confidence(result)
            result['conf'] = scores

            """
            Metadata stores information we might be interested in storing 
            about unique addresses. Does not store anything in memory.

            Data only stores (user, deposit, exchange) tuples. 

            This division is helpful as to prevent a huge file. In clustering, 
            for example, we do not care about metadata. 
            """
            metadata: Dict[str, Dict[str, Any]] = self._make_metadata(result)
            if chunk_count == 0:
                metadata.to_csv(metadata_file, index=False)
                result.to_csv(result_file, index=False)
            else:
                metadata.to_csv(metadata_file, mode='a', header=False, index=False)
                result.to_csv(result_file, mode='a', header=False, index=False)

            if (max_block - min_block <= self.t_max):
                print('Consider choosing a larger chunksize.')

            # this ensures we always have at least t_max
            # also provides some overlap
            last_chunk: pd.DataFrame = tx_chunk[
                tx_chunk.block_number >= (min_block - (self.t_max + 1))
            ].copy()

            print('.', end = '', flush=True)  # progress bar
            chunk_count += 1

            del metadata, result, tx_chunk, scores, both_chunks

    def _make_metadata(self, data: pd.DataFrame):
        """
        Store anything we may want to lookup about these people.

        Note this is slightly inefficient because we could get uniques
        but we need to preserve pairing between exchange and deposit.
        """
        metadata: Dict[str, Dict[str, Any]] = dict()

        metadata: Dict[str, List[str]] = {
            'address': [],
            'entity': [],
            'conf': [],
            'meta_data': [],
        }

        for user, user_df in data.groupby('user'):
            conf: float = user_df.conf.mean()
            metadata['address'].append(user)
            metadata['conf'].append(conf)
            metadata['entity'].append(Entity.EOA.value)
            metadata['meta_data'].append(json.dumps({}))

        for exchange, exchange_df in data.groupby('exchange'):
            conf: float = exchange_df.conf.mean()
            exchange_metadata: Dict[str, Any] = self.loader.get_exchange_metadata(exchange)
            metadata['address'].append(exchange)
            metadata['conf'].append(conf)
            metadata['entity'].append(Entity.EXCHANGE.value)
            metadata['meta_data'].append(json.dumps(exchange_metadata))

        for deposit, deposit_df in data.groupby('deposit'):
            exchange: str = deposit_df.iloc[0].exchange
            conf: float = deposit_df.conf.mean()
            exchange_metadata: Dict[str, Any] = self.loader.get_exchange_metadata(exchange)
            exchange_name: str = exchange_metadata['name']
            metadata['address'].append(deposit)
            metadata['conf'].append(conf)
            metadata['entity'].append(Entity.DEPOSIT.value)
            deposit_metadata: Dict[str, Any] = {
                'exchange_address': exchange,
                'exchange_name': exchange_name,
                'name': f'Deposit for {exchange_name}',
            }
            metadata['meta_data'].append(json.dumps(deposit_metadata))

        metadata: pd.DataFrame = pd.DataFrame(metadata)
        return metadata

    def _cluster_chunk(
        self,
        tx_chunk: pd.DataFrame,
        exchanges: pd.DataFrame,
        miners: pd.DataFrame,
        blacklist: pd.DataFrame,
    ) -> pd.DataFrame:
        exchange_addrs: np.array = exchanges.address
        blacklist_addrs: np.array = blacklist.address

        columns: List[str] = ['block_number', 'from_address', 'to_address', 'value']
        tx_chunk: pd.DataFrame = tx_chunk[columns].sort_values('block_number')
        tx_chunk['block'] = tx_chunk['block_number']  # dummy column

        # sender should not be a miner (avoid mining pools)
        tx_chunk: pd.DataFrame = tx_chunk[~tx_chunk['from_address'].isin(miners)]

        # check if receiver is an exchange
        is_exchange: pd.DataFrame = tx_chunk[tx_chunk['to_address'].isin(exchange_addrs)].copy()
        is_exchange['round_value'] = is_exchange['value'].round(1)

        # deposit addresses are those that send to exchanges
        deposits: pd.DataFrame = is_exchange['from_address']
        # deposits cannot be addresses inside the blacklist_addrs
        deposits: pd.DataFrame = deposits[~deposits.isin(blacklist_addrs)]

        # find addresses that send tokens to these deposits
        senders: pd.DataFrame = tx_chunk[tx_chunk['to_address'].isin(deposits)].copy()
        senders['round_value'] = senders['value'].round(1)
        # sender addresses (users) cannot be addresses inside the blacklist
        senders: pd.DataFrame = senders[~senders['from_address'].isin(blacklist_addrs)]

        deposit: pd.DataFrame = pd.merge_asof(
            left = is_exchange,
            right = senders,
            left_on = 'block',
            right_on = 'block',
            left_by = ['from_address', 'round_value'], 
            right_by = ['to_address', 'round_value'],
            tolerance = self.t_max, 
            direction = 'backward',
            allow_exact_matches=True,
            suffixes=['_y', '_x'],
        ).dropna()

        deposit['t_diff'] = deposit['block_number_y'] - deposit['block_number_x']
        deposit['a_diff'] = deposit['value_x'] - deposit['value_y']

        # round to ignore small errors
        deposit: pd.DataFrame = deposit[
            (deposit.a_diff.round(3) <= self.a_max) & 
            (deposit.a_diff.round(3) >= 0)
        ]

        # keep important columns and rename
        results: pd.DataFrame = deposit[
            ['from_address_x', 'from_address_y', 'to_address_y', 't_diff', 'a_diff']
        ]
        results.columns = ['user', 'deposit', 'exchange', 't_diff', 'a_diff']
        results: pd.DataFrame = results.drop_duplicates()

        return results

    def _get_confidence(
        self,
        tx_chunk: pd.DataFrame,
        time_weight: float = 1,
        amount_weight: float = 1,
    ) -> pd.DataFrame:
        """
        For the deposit reuse heuristic, the confidence of a cluster 
        lies solely on the confidence in an address being a deposit 
        address. This is determined by two measures in the algorithm: 
        (1) the difference in block times and (2) the different in
        transaction amounts. The larger of each, the lower confidence.
        That is, if both differences were 0, we should be fully confident
        in labeling a deposit address.

        A true probabilistic value is desirable but hard to find. We 
        make a linear assumption. Given a maximum value, we linearly 
        interpolate from 0 (full confidence) to that max (no confidence),
        and ascribe that as a confidence value.

        More sophisticated approximations might assume a higher degree
        polynomial or a distribution (e.g. beta) between scores and 
        confidence. We opt for the simplest for now.

        Algorithm:
        --
        1) compute t_s = |t1 - t2| / t_max  # time score
        2) compute a_s = |a1 - a2| / a_max  # amount score
        3) t_conf = 1 - t_s
        4) t_conf = 1 - a_s

        Note that this algorithm is not without hyperparameters. There 
        are two hyperparameters fitting the beta distribution.
        """
        t_score: pd.DataFrame = tx_chunk['t_diff'].abs() / self.t_max
        t_score: pd.DataFrame = t_score.clip(0, 1)

        a_score: pd.DataFrame = tx_chunk['a_diff'].abs() / self.a_max
        a_score: pd.DataFrame = a_score.clip(0, 1)

        def get_confidence(score):
            return 1 - score

        t_conf: pd.DataFrame = get_confidence(t_score)
        a_conf: pd.DataFrame = get_confidence(a_score)

        # Actual confidence is a weighted average
        conf: pd.DataFrame = (
            t_conf * time_weight + a_conf * amount_weight) / 2.

        return conf
