import os
from typing import Any, Iterable, Dict, Optional, Set
import pandas as pd


class DataLoader:
    """
    Inherit me for new data loaders.
    """

    def get_blocks(self) -> Iterable[Any]:
        raise NotImplementedError
    
    def yield_transactions(self, chunk_size: int = 10000) -> Iterable[Any]:
        raise NotImplementedError


class DataframeLoader(DataLoader):
    """
    Standard loader reading CSV files.

    Supports toy Bionic token data from etherclust:
    https://github.com/etherclust/etherclust/tree/master/data

    Supports processed BigQuery database.
    """

    def __init__(
        self,
        block_csv: str,
        known_addresses_csv: str,
        transaction_csv: str,
        cache_dir: str,
    ):
        print('fetching exchanges...')
        known_addresses: pd.DataFrame = pd.read_csv(known_addresses_csv)
        exchanges: pd.DataFrame = known_addresses[
            (known_addresses.account_type == 'eoa') & 
            (known_addresses.entity == 'exchange')
        ]
        print(f'found {len(exchanges)} exchanges.')

        # inverse of exchanges (everything else in known addresses should
        # not be considered an EOA or deposit.)
        blacklist: pd.DataFrame = known_addresses[
            (known_addresses.account_type != 'eoa') |
            (known_addresses.entity != 'exchange')
        ]
        print(f'found {len(blacklist)} addresses to ignore.')

        # find miners
        print('fetching miners...')
        miners_file: str = os.path.join(cache_dir, 'miners.csv')
        miners: pd.Series = self._find_miners(block_csv, cache_file = miners_file)

        self._exchanges: pd.DataFrame = exchanges
        self._miners: pd.Series = miners
        self._blacklist: pd.DataFrame = blacklist

        self._block_csv = block_csv
        self._transaction_csv = transaction_csv
        self._known_addresses_csv = known_addresses_csv

    def get_exchanges(self) -> pd.DataFrame:
        return self._exchanges

    def get_exchange_metadata(self, address: str) -> Dict[str, Any]:
        df: pd.DataFrame = self._exchanges[self._exchanges.address == address]
        if len(df) == 0: 
            return {}  # nothing to do
        metadata: Dict[str, Any] = df.iloc[0].to_dict()
        del metadata['address']  # don't need this
        return metadata

    def get_miners(self) -> pd.DataFrame:
        return self._miners

    def get_blacklist(self) -> pd.DataFrame:
        return self._blacklist

    def _find_miners(
        self,
        block_csv: str,
        chunk_size: int = 10000,
        cache_file: Optional[str] = None,
    ) -> pd.Series:
        """
        Load a segment of the block csv at a time and store unique
        miner addresses. Otherwise, it is too large.
        """
        if os.path.isfile(cache_file):
            miners = pd.read_csv(cache_file)
        else:
            miners: Set = set()
            for chunk in pd.read_csv(block_csv, chunksize = chunk_size):
                chunk: pd.DataFrame = chunk
                chunk_miners: pd.DataFrame = chunk.miner.drop_duplicates()
                chunk_miners: Set = set(chunk_miners)
                miners: Set = miners.union(chunk_miners)
            miners: pd.Series = pd.Series(list(miners))
            
            cache_dir: str = os.path.dirname(cache_file)
            if not os.path.isdir(cache_dir): os.makedirs(cache_dir)
            miners.to_csv(cache_file)
        print(f'found {len(miners)} miners.')
        return miners

    def yield_transactions(
        self,
        chunk_size: int = 10000,
    ) -> Iterable[pd.DataFrame]:
        """
        Load a segment at a time (otherwise too large).
        """
        for chunk in pd.read_csv(self._transaction_csv, chunksize = chunk_size):
            yield chunk
