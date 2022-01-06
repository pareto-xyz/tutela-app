"""
Contains utilities to download recent data needed to run heuristics
on tornado cash data.

This script relies on `bq`. Make sure that this is setup on the instance
running this script.

Most of the heuristics require access to all of tornado cash at once.
As such, we cannot just download a daily slice (an increment) and 
process that. The best we can do is the following:

1) Keep a table of Tornado Cash trace and transaction data in BigQuery. 
We will update this table daily. 

2) Once updated, the table will be exported a Google cloud bucket. We 
will delete prior contents in bucket before exporting.

3) Download the trace and transactions buckets locally and rename to 
expected tornado_trace.csv and tornado_transaction.csv.

4) Run the preprocessing script to generate complete_withdraw_tx.csv
and complete_deposit_tx.csv. 

5) Save to disk (overwrite old files).
"""
import os
import sys
import pandas as pd
from os.path import join
from typing import Tuple, Optional, List, Dict, Any

from live import utils
from src.tcash.data import decode_transactions


def get_last_block():
    """
    Read the current transactions dataframe to see what the latest block is.
    We will grab all data from the block and after.
    """
    data_path:  str = utils.CONSTANTS['data_path']
    cache_path: str = join(data_path, 'live/tornado_cash/tornado_cache')
    transactions: pd.DataFrame = pd.read_csv(join(cache_path, 'tornado_transactions.csv'))
    last_block: int = int(transactions.block_number.max())

    return last_block


def update_bigquery(start_block: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Run SQL queries against BigQuery to insert the most recent data into
    the following tables.

      tornado_transactions.traces
      tornado_transactions.transactions

    We assume your bigquery project has a `tornado_transactions` dataset
    with the two tables already existing. If not, please make them prior
    to running this script.

    We intentionally use `bq` instead of the Python BigQuery library as
    `bq` is orders of magnitudes faster.
    """
    project: str = utils.CONSTANTS['bigquery_project']
    bq_trace: str = 'bigquery-public-data.crypto_ethereum.traces'
    bq_transaction: str = 'bigquery-public-data.crypto_ethereum.transactions'
    contract_sql: str = f'select address from {project}.tornado_transactions.tornadocontracts'
    subtrace_sql: str = f'select transaction_hash from {project}.tornado_transactions.traces'

    trace_query: str = make_bq_query(
        f'select * from {bq_trace}',
        where_clauses = [
            f'to_address in ({contract_sql})',
            'substr(input, 1, 10) in ("0xb214faa5", "0x21a0adb6")',
            f'block_number > {start_block}',
        ],
        flags = [
            f'--destination_table {project}:tornado_transactions.traces',
            '--use_legacy_sql=false',
        ],
    )
    transaction_query: str = make_bq_query(
        f'select * from {bq_transaction} as b',
        where_clauses = [
            f'b.hash in ({subtrace_sql})',
            f'b.block_number > {start_block}',
        ],
        flags = [
            f'--destination_table {project}:tornado_transactions.transactions',
            '--use_legacy_sql=false',
        ],
    )
    trace_success: bool = utils.execute_bash(trace_query)
    transaction_success: bool = utils.execute_bash(transaction_query)

    success: bool = trace_success and transaction_success
    return success, {}


def make_bq_query(
    select: str, where_clauses: List[str] = [], flags: List[str] = []) -> str:
    flags: str = ' '.join(flags)
    where_clauses: List[str] = [f'({clause})' for clause in where_clauses]
    where_clauses: str = ' and '.join(where_clauses)
    query: str = f"bq_query {flags} '{select} where {where_clauses}'"
    return query


def empty_bucket() -> Tuple[bool, Dict[str, Any]]:
    """
    Make sure nothing is in bucket (we want to overwrite).
    """
    trace_success: bool = utils.delete_bucket_contents('tornado-trace')
    transaction_success: bool = utils.delete_bucket_contents('tornado-transaction')

    success: bool = trace_success and transaction_success
    return success, {}


def update_bucket() -> Tuple[bool, Dict[str, Any]]:
    """
    Move the updated bigquery data to bucket.
    """
    project: str = utils.CONSTANTS['bigquery_project']
    trace_success: bool = utils.export_bigquery_table_to_cloud_bucket(
        f'{project}.tornado_transactions',
        'traces',
        'tornado-trace',
    )
    transaction_success: bool = utils.export_bigquery_table_to_cloud_bucket(
        f'{project}.tornado_transactions',
        'transactions',
        'tornado-transaction',
    )
    success: bool = trace_success and transaction_success

    return success, {}


def download_bucket() -> Tuple[bool, Any]:
    """
    Make sure nothing is in bucket (we want to overwrite).
    """
    data_path:  str = utils.CONSTANTS['data_path']
    out_dir = join(data_path, 'live/tornado_cash')
    trace_success, trace_files = utils.export_cloud_bucket_to_csv('tornado-trace', out_dir)
    transaction_success, transaction_files = utils.export_cloud_bucket_to_csv('tornado-transaction', out_dir)

    success: bool = trace_success and transaction_success
    data = {'trace': trace_files, 'transaction': transaction_files}

    return success, data


def get_deposit_and_withdraw(
    trace_df: pd.DataFrame, 
    transaction_df: pd.DataFrame) -> Tuple[bool, Dict[str, pd.DataFrame]]:
    data_path:  str = utils.CONSTANTS['data_path']
    contract_dir = join(data_path, 'static/tcash')

    try:
        address_df: pd.DataFrame = pd.read_csv(
            join(contract_dir, 'tornado_contract_abi.csv'),
            names=['address', 'token', 'value', 'name','abi'],
            sep='|')
        proxy_df = pd.read_csv(
            join(contract_dir, 'tornado_proxy_abi.csv'), 
            names=['address', 'abi'],
            sep='|')
        deposit_df, withdraw_df = decode_transactions(
            address_df, proxy_df, transaction_df, trace_df)
        success: bool = True
        data = {'withdraw': withdraw_df, 'deposit': deposit_df}
    except:
        success: bool = False
        data: Dict[str, pd.DataFrame] = {}
    return success, data


def cache_merged_file(df: pd.DataFrame, name: str):
    data_path:  str = utils.CONSTANTS['data_path']
    out_dir = join(data_path, 'live/tornado_cash')
    df.to_csv(join(out_dir, name), index=False)


def delete_files(paths: List[str]):
    for path in paths:
        if os.path.isdir(path):
            os.remove(path)


def main():
    log_path: str = utils.CONSTANTS['log_path']
    os.makedirs(log_path, exist_ok=True)

    log_file: str = join(log_path, 'tornadocash.log')
    os.remove(log_file)  # remove old file (yesterday's)

    logger = utils.get_logger(log_file)

    logger.info('entering get_last_block')
    last_block: int = get_last_block()
    logger.info(f'last_block={last_block}')

    logger.info('entering update_bigquery')
    success, _ = update_bigquery(last_block)

    if not success:
        logger.error('failed on updating bigquery tables')
        sys.exit(0)

    logger.info('entering empty_bucket')
    success, _ = empty_bucket()

    if not success:
        logger.error('failed on emptying cloud buckets')
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

    trace_files: List[str] = data['trace']
    transaction_files: List[str] = data['transaction']

    if len(trace_files) == 0:
        logger.error('found 0 files for tornado cash traces')
        sys.exit(0)

    if len(transaction_files) == 0:
        logger.error('found 0 files for tornado cash transactions')
        sys.exit(0)

    logger.info('sorting and combining trace files')
    trace_df: pd.DataFrame = utils.load_data_from_chunks(trace_files)
    logger.info('sorting and combining transaction files')
    transaction_df: pd.DataFrame = utils.load_data_from_chunks(transaction_files)

    logger.info('deleting trace chunks')
    cache_merged_file(trace_df, 'tornado_traces.csv')
    logger.info('deleting transaction chunks')
    cache_merged_file(transaction_df, 'tornado_transactions.csv')

    delete_files(trace_files)
    delete_files(transaction_files)

    logger.info('entering get_deposit_and_withdraw')
    success, data = get_deposit_and_withdraw(trace_df, transaction_df)

    if not success:
        logger.error('failed on computing deposit and withdraw dataframes')
        sys.exit(0)

    deposit_df: pd.DataFrame = data['deposit']
    withdraw_df: pd.DataFrame = data['withdraw']

    # TODO: run heuristics on these dataframes


if __name__ == "__main__":
    main()
