import os, sys
import psycopg2
import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from typing import Tuple, Optional, List, Dict, Any

from live import utils
from live.bq_utils import make_bq_query, make_bq_delete


def get_last_block():
    """
    Read the current transactions dataframe to see what the latest block is.
    We will grab all data from the block and after.
    """
    data_path:  str = utils.CONSTANTS['data_path']
    depo_path: str = join(data_path, 'live/depositreuse')
    transactions: pd.DataFrame = pd.read_csv(
        join(depo_path, 'ethereum_transactions_live.csv'))
    last_block: int = int(transactions.block_number.max())

    return last_block


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

    bq_block: str = 'bigquery-public-data.crypto_ethereum.blocks'
    bq_transaction: str = 'bigquery-public-data.crypto_ethereum.transactions'
    block_table: str = 'crypto_ethereum.blocks_live'
    transaction_table: str = 'crypto_ethereum.transactions_live'

    flags: List[str] = ['--use_legacy_sql=false']

    if delete_before:
        block_init: str = make_bq_delete(block_table, flags = flags)
        block_init_success: bool = utils.execute_bash(block_init)

        transaction_init: str = make_bq_delete(transaction_table, flags = flags)
        transaction_init_success: bool = utils.execute_bash(transaction_init)
    else:  # nothing to do
        block_init_success: bool = True
        transaction_init_success: bool = True

    init_success = block_init_success and transaction_init_success

    if not init_success:
        return init_success, {}

    block_columns: List[str] = [
        'hash as block_hash', 
        'miner', 
        'timestamp', 
        'number',
    ]
    block_columns: List[str] = [f'b.{col}' for col in block_columns]
    block_columns: str = ','.join(block_columns)
    block_select_sql: str = f"select {block_columns} from {bq_block} as b"
    block_query: str = make_bq_query(
        f'insert into {project}.{block_table} {block_select_sql}',
        where_clauses = [
            f'b.number > {start_block}',
        ],
        flags = flags,
    )
    block_query_success: bool = utils.execute_bash(block_query)

    transaction_columns: List[str] = [
        'hash as transaction', 
        'from_address', 
        'to_address', 
        'value', 
        'block_timestamp', 
        'block_number',
    ]
    transaction_columns: List[str] = [f'b.{col}' for col in transaction_columns]
    transaction_columns: str = ','.join(transaction_columns)
    transaction_select_sql: str = f"select {transaction_columns} from {bq_transaction} as b"
    transaction_query: str = make_bq_query(
        f'insert into {project}.{transaction_table} {transaction_select_sql}',
        where_clauses = [
            f'b.block_number > {start_block}',
        ],
        flags = flags,
    )
    transaction_query_success: bool = utils.execute_bash(transaction_query)

    success: bool = block_query_success and transaction_query_success
    return success, {}


def empty_bucket() -> Tuple[bool, Dict[str, Any]]:
    """
    Make sure nothing is in bucket (we want to overwrite).
    """
    block_success: bool = utils.delete_bucket_contents('ethereum-block-data-live')
    transaction_success: bool = utils.delete_bucket_contents('ethereum-transaction-data-live')
    success: bool = block_success and transaction_success
    return success, {}


def update_bucket() -> Tuple[bool, Dict[str, Any]]:
    project: str = utils.CONSTANTS['bigquery_project']
    block_success: bool = utils.export_bigquery_table_to_cloud_bucket(
        f'{project}.crypto_ethereum',
        'blocks_live',
        'ethereum-block-data-live',
    )
    transaction_success: bool = utils.export_bigquery_table_to_cloud_bucket(
        f'{project}.crypto_ethereum',
        'transactions_live',
        'ethereum-transaction-data-live',
    )
    success: bool = block_success and transaction_success
    return success, {}


def download_bucket() -> Tuple[bool, Any]:
    data_path:  str = utils.CONSTANTS['data_path']
    out_dir = join(data_path, 'live/depositreuse')
    block_success, block_files = utils.export_cloud_bucket_to_csv(
        'ethereum-block-data-live', out_dir)
    transaction_success, transaction_files = utils.export_cloud_bucket_to_csv(
        'ethereum-transaction-data-live', out_dir)
    data = {
        'block': block_files,
        'transaction': transaction_files,
    }
    success: bool = block_success and transaction_success
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
    # log_path: str = utils.CONSTANTS['log_path']
    # os.makedirs(log_path, exist_ok=True)

    # log_file: str = join(log_path, 'depositreuse-data.log')
    # if os.path.isfile(log_file):
    #     os.remove(log_file)  # remove old file (yesterday's)

    # logger = utils.get_logger(log_file)

    # if args.scratch:
    #     logger.info('starting from scratch')
    #     # NOTE: scratch here does not mean starting from 0, that would be
    #     # ridiculous, this instead will be starting from our known endpt.
    #     # This data is already loaded into our db. We are even writing to 
    #     # a separate table/bucket to preserve this checkpoint.
    #     last_block: int = 13330090
    # else:
    #     logger.info('entering get_last_block')
    #     last_block: int = get_last_block()
    #     logger.info(f'last_block={last_block}')

    # logger.info('entering update_bigquery')
    # # NOTE: we always wipe this table. This bigquery table ONLY stores 
    # # the most recent data.
    # success, _ = update_bigquery(last_block, delete_before = True)

    # if not success:
    #     logger.error('failed on updating bigquery tables')
    #     sys.exit(0)

    # logger.info('entering empty_bucket')
    # success, _ = empty_bucket()

    # if not success:
    #     logger.error('failed on emptying cloud buckets')
    #     sys.exit(0)

    # logger.info('entering update_bucket')
    # success, _ = update_bucket()

    # if not success:
    #     logger.error('failed on updating cloud buckets')
    #     sys.exit(0)

    # logger.info('entering download_bucket')
    # success, data = download_bucket()

    # if not success:
    #     logger.error('failed on downloading cloud buckets')
    #     sys.exit(0)

    # block_files: List[str] = data['block']
    # transaction_files: List[str] = data['transaction']
    
    root: str = utils.CONSTANTS['data_path']
    block_files = glob(
        join(root, 'live/depositreuse/blocks_live-*.csv'))
    transaction_files = glob(
        join(root, 'live/depositreuse/transactions_live-*.csv'))

    if len(block_files) == 0:
        # logger.error('found 0 files for deposit reuse blocks')
        sys.exit(0)
    
    if len(transaction_files) == 0:
        # logger.error('found 0 files for deposit reuse transactions')
        sys.exit(0)

    # logger.info('sorting and combining block files')
    block_df: pd.DataFrame = utils.load_data_from_chunks(
        block_files, sort_column = 'number')
    block_df.drop_duplicates('hash', inplace=True)

    # logger.info('saving block chunks')
    save_file(block_df, 'ethereum_blocks_live.csv')

    # logger.info('deleting block files')
    delete_files(block_files)
    
    # logger.info('sorting and combining transaction files')
    transaction_out_file: str = join(
        utils.CONSTANTS['data_path'], 
        'live/depositreuse/ethereum_transactions_live.csv')
    
    utils.load_data_from_chunks_low_memory(
        transaction_files, 
        transaction_out_file,
        sort_column = 'block_number')

    # logger.info('deleting transaction files')
    delete_files(transaction_files)


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
