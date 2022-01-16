import pandas as pd
from io import StringIO
from typing import List
from collections import deque


def get_header(csv_file: str) -> List[str]:
    return pd.read_csv(csv_file, index_col=0, nrows=0).columns.tolist()


def restore_last_chunk(
    transaction_csv: str,
    chunk_csv: str, 
    min_block: int,
    t_max: int = 3200) -> pd.DataFrame:
    """
    In the deposit reuse algorithm, it is important to get a last chunk.
    We will want to store a live version of this locally but that if something
    should happen and either this saved copy goes missing or is outdated, we 
    need an easy way to produce a new one.

    We need to do this efficiently, 

    @transaction_csv: (str) path to most recent transaction file available.
    @chunk_size: (int) how big of a chunk we want
    @chunk_csv: (str) where to save the file
    """
    header: List[str] = get_header(transaction_csv)

    last_chunk: pd.DataFrame = []
    for chunk in pd.read_csv(transaction_csv, chunksize=10000*t_max):
        part = chunk[chunk.block_number >= (min_block - (t_max + 1))]
        last_chunk.append(part)
        if len(part) > 0:
            print('!', end = '', flush=True)
        else:
            print('.', end = '', flush=True)
        del chunk

    last_chunk = pd.concat(last_chunk)
    last_chunk.to_csv(chunk_csv, index=False)
