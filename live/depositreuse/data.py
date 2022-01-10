import os, sys
import psycopg2
import numpy as np
import pandas as pd
from os.path import join
from typing import Tuple, Optional, List, Dict, Any

from live import utils
from live.bq_utils import make_bq_query, make_bq_delete


def update_bigquery(
    start_block: int,
    delete_before: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Run SQL queries against BigQuery to insert the most recent data into
    the following tables.

        crypto_ethereum.transactions

    This will be a very big table. Generally, we will only be adding rows
    into this table. For the first time, we may wish to run from scratch 
    although we do not recommend this. 

    We intentionally use `bq` instead of the Python BigQuery library as
    `bq` is orders of magnitudes faster.
    """
    project: str = utils.CONSTANTS['bigquery_project']
    bq_transaction: str = 'bigquery-public-data.crypto_ethereum.transactions'
    transaction_table: str = 'crypto_ethereum.transactions_live'

    flags: List[str] = ['--use_legacy_sql=false']

    if delete_before:
        init: str = make_bq_delete(transaction_table, flags = flags)
        init_success: bool = utils.execute_bash(init)
    else:  # nothing to do
        init_success: bool = True
    
    columns: List[str] = [
        'from_address', 
        'to_address', 
        'hash as transaction', 
        'value', 
        'block_timestamp', 
        'block_number',
    ]
    columns: List[str] = [f'b.{col}' for col in columns]
    columns: str = ','.join(columns)
    select_sql: str = f"{columns} from {bq_transaction} as b"
    query: str = make_bq_query(
        f'insert into {project}.{transaction_table} {select_sql}',
        where_clauses = [
            f'b.block_number <= {start_block}',
        ],
        flags = flags,
    )
    query_success: bool = utils.execute_bash(query)

    success: bool = init_success and query_success
    return success, {}


def update_bucket() -> Tuple[bool, Dict[str, Any]]:
    project: str = utils.CONSTANTS['bigquery_project']
    success: bool = utils.export_bigquery_table_to_cloud_bucket(
        f'{project}.crypto_ethereum',
        'transactions_live',
        'ethereum-transaction-data-live',
    )
    return success, {}


def download_bucket() -> Tuple[bool, Any]:
    data_path:  str = utils.CONSTANTS['data_path']
    out_dir = join(data_path, 'live/depositreuse')
    success, files = utils.export_cloud_bucket_to_csv('ethereum-transaction-data-live', out_dir)
    data = {'transaction': files}
    return success, data


def save_file(df: pd.DataFrame, name: str):
    data_path:  str = utils.CONSTANTS['data_path']
    out_dir: str = join(data_path, 'live/depositreuse')
    out_file: str = join(out_dir, name)
    df.to_csv(out_file, index=False)
    return out_file


def delete_files(paths: List[str]):
    for path in paths:
        if os.path.isfile(path):
            os.remove(path)


def main(args: Any):
    log_path: str = utils.CONSTANTS['log_path']
    os.makedirs(log_path, exist_ok=True)

    log_file: str = join(log_path, 'depositreuse-data.log')
    if os.path.isfile(log_file):
        os.remove(log_file)  # remove old file (yesterday's)

    logger = utils.get_logger(log_file)

    if args.scratch:
        logger.info('starting from scratch')
        last_block: int = 13330090
    else:
        logger.info('entering get_last_block')
        last_block: int = get_last_block()
        logger.info(f'last_block={last_block}')

    logger.info('entering update_bigquery')
    success, _ = update_bigquery(last_block)

    if not success:
        logger.error('failed on updating bigquery tables')
        sys.exit(0)

    logger.info('entering update_bucket')
    success, _ = update_bucket()

    if not success:
        logger.error('failed on updating cloud buckets')
        sys.exit(0)

    logger.info('entering download_bucket')
    success, data = download_bucket()

    if not success:
        logger.error('failed on downloading cloud buckets')
        sys.exit(0)

    files: List[str] = data['transaction']
    if len(files) == 0:
        logger.error('found 0 files for deposit reuse transactions')
        sys.exit(0)

    logger.info('sorting and combining trace files')
    df: pd.DataFrame = utils.load_data_from_chunks(files)
    df.drop_duplicates('transaction', inplace=True)

    logger.info('saving transaction chunks')
    save_file(df, 'ethereum_transactions_live.csv')

    logger.info('deleting trace files')
    delete_files(files)


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', action='store_true', default=False)
    parser.add_argument('--no-db', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
