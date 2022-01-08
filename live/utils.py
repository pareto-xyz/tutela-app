import logging
import subprocess
import pandas as pd
from typing import Tuple, List
from os.path import join, dirname, realpath

from src.utils.bigquery import EthereumBigQuery
from src.utils.storage import EthereumStorage

LIVE_DIR: str = realpath(dirname(__file__))
ROOT_DIR: str = realpath(join(LIVE_DIR, '..'))
DATA_DIR: str = realpath(join(ROOT_DIR, 'data'))
LOG_DIR: str = realpath(join(ROOT_DIR, 'logs'))

CONSTANTS = {
    'live_path': LIVE_DIR,
    'root_path': ROOT_DIR,
    'data_path': DATA_DIR,
    'log_path': LOG_DIR,
    'bigquery_project': 'lexical-theory-329617',
    'postgres_db': 'tornado',
    'postgres_user': 'postgres',
}


def export_bigquery_table_to_cloud_bucket(
    src_dataset: str,
    src_table: str,
    dest_bucket: str,
) -> bool:
    handler: EthereumBigQuery = EthereumBigQuery(src_dataset)
    try:
        handler.export_to_bucket(src_table, dest_bucket)
        return True
    except: 
        return False


def delete_bucket_contents(bucket: str) -> bool:
    handler: EthereumStorage = EthereumStorage()
    try:
        handler.empty_bucket(bucket)
        return True
    except: 
        return False


def export_cloud_bucket_to_csv(
    bucket: str, out_dir: str) -> Tuple[bool, List[str]]:
    handler: EthereumStorage = EthereumStorage()
    try:
        files: List[str] = handler.export_to_csv(bucket, out_dir)
        return True, files
    except: 
        return False, []


def execute_bash(cmd: str) -> bool:
    code: int = subprocess.call(cmd, shell=True)
    return code == 0


def get_logger(log_file: str) -> logging.basicConfig:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the file handler to the logger
    logger.addHandler(handler)

    return logger


def load_data_from_chunks(files: List[str]) -> pd.DataFrame:
    """
    Read dataframes from files and sort by block number.
    """
    chunks: List[pd.DataFrame] = []
    for file_ in files:
        chunk: pd.DataFrame = pd.read_csv(file_)
        chunks.append(chunk)

    data: pd.DataFrame = pd.concat(chunks)
    data.reset_index(inplace=True)
    data.sort_values('block_number', inplace=True)

    return data
