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
import psycopg2
import numpy as np
import pandas as pd
from os.path import join
from typing import Tuple, Optional, List, Dict, Any

from live import utils
from src.tcash.data import decode_transactions
from src.utils.utils import from_json


def get_last_block():
    """
    Read the current transactions dataframe to see what the latest block is.
    We will grab all data from the block and after.
    """
    data_path:  str = utils.CONSTANTS['data_path']
    tcash_path: str = join(data_path, 'live/tornado_cash')
    transactions: pd.DataFrame = pd.read_csv(join(tcash_path, 'tornado_transactions.csv'))
    last_block: int = int(transactions.block_number.max())

    return last_block


def update_bigquery(start_block: int) -> Tuple[bool, Dict[str, Any]]:
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

    trace_table: str = 'tornado_transactions.traces'
    transaction_table: str = 'tornado_transactions.transactions'
    miner_table: str = 'tornado_transactions.miner_transactions'

    flags: List[str] = ['--use_legacy_sql=false']

    trace_prepare: str = make_bq_delete(trace_table, flags = flags)
    trace_query: str = make_bq_query(
        f'insert into {project}.{trace_table} select * from {bq_trace}',
        where_clauses = [
            f'to_address in ({contract_sql})',
            'substr(input, 1, 10) in ("0xb214faa5", "0x21a0adb6")',
            f'block_number > {start_block}',
        ],
        flags = flags,
    )
    transaction_prepare: str = make_bq_delete(transaction_table, flags = flags)
    transaction_query: str = make_bq_query(
        f'insert into {project}.{transaction_table} select * from {bq_transaction} as b',
        where_clauses = [
            f'b.hash in ({subtrace_sql})',
            f'b.block_number > {start_block}',
        ],
        flags = flags,
    )

    # This is for the TORN mining heuristic -- we need to get miner's txs
    miner_prepare: str = make_bq_delete(miner_table, flags = flags)
    miner_query: str = make_bq_query(
        f'insert into {project}.{miner_table} select * from {bq_transaction}',
        where_clauses = [
            'to_address = "0x746aebc06d2ae31b71ac51429a19d54e797878e9"',
            f'block_number > {start_block}',
        ],
        flags = flags,
    )
    trace0_success: bool = utils.execute_bash(trace_prepare)
    trace1_success: bool = utils.execute_bash(trace_query)
    transaction0_success: bool = utils.execute_bash(transaction_prepare)
    transaction1_success: bool = utils.execute_bash(transaction_query)
    miner0_success: bool = utils.execute_bash(miner_prepare)
    miner1_success: bool = utils.execute_bash(miner_query)

    success: bool = (trace0_success and trace1_success) and \
                    (transaction0_success and transaction1_success) and \
                    (miner0_success and miner1_success)
    return success, {}


def make_bq_delete(table: str, flags: List[str]) -> str:
    project: str = utils.CONSTANTS['bigquery_project']
    flags: str = ' '.join(flags)
    statement: str = f'delete from {project}.{table} where true'
    query: str = f"bq query {flags} '{statement}'"
    return query


def make_bq_query(
    select: str, where_clauses: List[str] = [], flags: List[str] = []) -> str:
    flags: str = ' '.join(flags)
    where_clauses: List[str] = [f'({clause})' for clause in where_clauses]
    where_clauses: str = ' and '.join(where_clauses)
    query: str = f"bq query {flags} '{select} where {where_clauses}'"
    return query


def make_bq_load(table: str, csv_path: str, schema: str) -> str:
    project: str = utils.CONSTANTS['bigquery_project']
    flags: List[str] = [
        "--skip_leading_rows=1",
        "--field_delimiter='\t'",
        "--source_format=CSV",
    ]
    flags: str = ' '.join(flags)
    command: str = f"bq load {flags} {project}:{table} {csv_path} {schema}"
    return command


def empty_bucket() -> Tuple[bool, Dict[str, Any]]:
    """
    Make sure nothing is in bucket (we want to overwrite).
    """
    trace_success: bool = utils.delete_bucket_contents('tornado-trace')
    transaction_success: bool = utils.delete_bucket_contents('tornado-transaction')
    miner_success: bool = utils.delete_bucket_contents('tornado-miner-transaction')

    success: bool = trace_success and transaction_success and miner_success
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
    miner_success: bool = utils.export_bigquery_table_to_cloud_bucket(
        f'{project}.tornado_transactions',
        'miner_transactions',
        'tornado-miner-transaction',
    )
    success: bool = trace_success and transaction_success and miner_success

    return success, {}


def download_bucket() -> Tuple[bool, Any]:
    """
    Make sure nothing is in bucket (we want to overwrite).
    """
    data_path:  str = utils.CONSTANTS['data_path']
    out_dir = join(data_path, 'live/tornado_cash')
    trace_success, trace_files = utils.export_cloud_bucket_to_csv('tornado-trace', out_dir)
    transaction_success, transaction_files = utils.export_cloud_bucket_to_csv('tornado-transaction', out_dir)
    miner_success, miner_files = utils.export_cloud_bucket_to_csv('tornado-miner-transaction', out_dir)

    success: bool = trace_success and transaction_success and miner_success
    data = {'trace': trace_files, 'transaction': transaction_files, 'miner': miner_files}

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


def external_pipeline(
    start_block: int, 
    deposit_df: pd.DataFrame, 
    withdraw_df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    We need to update another bigquery table for external transactions
    between TornadoCash users. We must do this separately because we 
    need `deposit_df` and `withdraw_df`. 

    This function will also move the table to a google bucket and 
    download that bucket locally, combine, sort, and save.
    """
    deposit_addresses: List[str] = deposit_df.from_address.unique().tolist()
    withdraw_addresses: List[str] = withdraw_df.recipient_address.unique().tolist()

    deposit_address_df: pd.DataFrame = pd.DataFrame.from_dict(
        {'address': deposit_addresses})
    withdraw_address_df: pd.DataFrame = pd.DataFrame.from_dict(
        {'address': withdraw_addresses})

    deposit_file: str = save_file(deposit_address_df, 'deposit_addrs.csv')
    withdraw_file: str = save_file(withdraw_address_df, 'withdraw_addrs.csv')

    # upload files to bigquery
    flags: List[str] = ['--use_legacy_sql=false']
    deposit_address_table: str = 'tornado_transactions.deposit_addresses'
    withdraw_address_table: str = 'tornado_transactions.withdraw_addresses'
    deposit_prepare: str = make_bq_delete(deposit_address_table, flags = flags)
    deposit_query: str = make_bq_load(deposit_address_table, deposit_file, 'address:string')
    withdraw_prepare: str = make_bq_delete(withdraw_address_table, flags = flags)
    withdraw_query: str = make_bq_load(withdraw_address_table, withdraw_file, 'address:string')
    deposit0_success: bool = utils.execute_bash(deposit_prepare)
    deposit1_success: bool = utils.execute_bash(deposit_query)
    withdraw0_success: bool = utils.execute_bash(withdraw_prepare)
    withdraw1_success: bool = utils.execute_bash(withdraw_query)
    success: bool = (deposit0_success and deposit1_success) and \
                    (withdraw0_success and withdraw1_success)

    if not success:
        return success, {}

    project: str = utils.CONSTANTS['bigquery_project']
    external_table: str = 'tornado_transactions.external_transactions'

    insert: str = f'insert {project}.{external_table}'
    select: str = 'select * from bigquery-public-data.crypto_ethereum.transactions'
    deposit_select: str = f'select address from {project}.tornado_transactions.deposit_addresses'
    withdraw_select: str = f'select address from {project}.tornado_transactions.withdraw_addresses'
    where_clauses: List[str] = [
        f'(from_address in ({deposit_select})) and (to_address in ({withdraw_select}))',
        f'(from_address in ({withdraw_select})) and (to_address in ({deposit_select}))',
        f'block_number > {start_block}',
    ]
    where_clauses: str = ' or '.join(where_clauses)
    query: str = f"bq query {' '.join(flags)} '{insert} {select} where {where_clauses}'"

    prepare: str = make_bq_delete(external_table, flags = flags)
    success0: bool = utils.execute_bash(prepare)
    success1: bool = utils.execute_bash(query)

    if not (success0 and success1):
        return False, {}

    # now move to google cloud bucket
    project: str = utils.CONSTANTS['bigquery_project']
    success: bool = utils.export_bigquery_table_to_cloud_bucket(
        f'{project}.tornado_transactions',
        'external_transactions',
        'tornado-external-transaction',
    )
    if not success:
        return success, {}

    # download google cloud bucket
    data_path:  str = utils.CONSTANTS['data_path']
    out_dir = join(data_path, 'live/tornado_cash')
    success, files = utils.export_cloud_bucket_to_csv('tornado-external-transaction', out_dir)

    if not success:
        return success, {}

    external_df: pd.DataFrame = utils.load_data_from_chunks(files)
    delete_files(files)

    save_file(external_df, 'external_txs.csv')


def save_file(df: pd.DataFrame, name: str):
    data_path:  str = utils.CONSTANTS['data_path']
    out_dir: str = join(data_path, 'live/tornado_cash')
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

    log_file: str = join(log_path, 'tornadocash-data.log')
    if os.path.isfile(log_file):
        os.remove(log_file)  # remove old file (yesterday's)

    logger = utils.get_logger(log_file)

    if args.scratch:
        logger.info('starting from scratch')
        last_block: int = 0
    else:
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
    miner_files: List[str] = data['miner']

    if len(trace_files) == 0:
        logger.error('found 0 files for tornado cash traces')
        sys.exit(0)

    if len(transaction_files) == 0:
        logger.error('found 0 files for tornado cash transactions')
        sys.exit(0)

    if len(miner_files) == 0:
        logger.error('found 0 files for tornado cash miners')
        sys.exit(0)

    logger.info('sorting and combining trace files')
    trace_df: pd.DataFrame = utils.load_data_from_chunks(trace_files)
    logger.info('sorting and combining transaction files')
    transaction_df: pd.DataFrame = utils.load_data_from_chunks(transaction_files)
    logger.info('sorting and combining miner files')
    miner_df: pd.DataFrame = utils.load_data_from_chunks(miner_files)

    # drop duplicates
    trace_df.drop_duplicates('transaction_hash', inplace=True)
    transaction_df.drop_duplicates('hash', inplace=True)
    miner_df.drop_duplicates('hash', inplace=True)

    logger.info('deleting trace chunks')
    save_file(trace_df, 'tornado_traces.csv')
    logger.info('deleting transaction chunks')
    save_file(transaction_df, 'tornado_transactions.csv')
    logger.info('deleting miner chunks')
    save_file(miner_df, 'miner_txs.csv')

    delete_files(trace_files)
    delete_files(transaction_files)
    delete_files(miner_files)

    logger.info('entering get_deposit_and_withdraw')
    success, data = get_deposit_and_withdraw(trace_df, transaction_df)

    if not success:
        logger.error('failed on computing deposit and withdraw dataframes')
        sys.exit(0)

    deposit_df: pd.DataFrame = data['deposit']
    withdraw_df: pd.DataFrame = data['withdraw']

    logger.info('entering external_pipeline')
    success, _ = external_pipeline(last_block, deposit_df, withdraw_df)

    if not success:
        logger.error('failed on processing external transactions')
        sys.exit(0)

    deposit_file: str = save_file(deposit_df, 'deposit_txs.csv')
    withdraw_file: str = save_file(withdraw_df, 'withdraw_txs.csv')

    if not args.no_db:
        # write csvs into databases
        conn = psycopg2.connect(
            database = utils.CONSTANTS['postgres_db'], 
            user = utils.CONSTANTS['postgres_user'],
        )
        cursor = conn.cursor()

        deposit_columns: List[str] = [
            'hash', 'transaction_index', 'from_address', 'to_address', 'gas',
            'gas_price', 'block_number', 'block_hash', 'tornado_cash_address'
        ]
        deposit_columns: str = ','.join(deposit_columns)
        command = f"COPY tornado_deposit({deposit_columns}) FROM '{deposit_file}' DELIMITER ',' CSV HEADER;"
        cursor.execute(command)

        withdraw_columns: List[str] = [
            'hash', 'transaction_index', 'from_address', 'to_address', 'gas',
            'gas_price', 'block_number', 'block_hash', 'tornado_cash_address',
            'recipient_address',
        ]
        withdraw_columns: str = ','.join(withdraw_columns)
        command: str = f"COPY tornado_withdraw({withdraw_columns}) FROM '{withdraw_file}' DELIMITER ',' CSV HEADER;"
        cursor.execute(command)

        conn.commit()

        cursor.close()
        conn.close()


if __name__ == "__main__":
    # import argparse 
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--scratch', action='store_true', default=False)
    # parser.add_argument('--no-db', action='store_true', default=False)
    # args = parser.parse_args()
    # main(args)

    data_path:  str = utils.CONSTANTS['data_path']
    tcash_path: str = join(data_path, 'live/tornado_cash')
    withdraw_df = pd.read_csv(join(tcash_path, 'withdraw_txs.csv'))
    deposit_df = pd.read_csv(join(tcash_path, 'deposit_txs.csv'))
    success, _ = external_pipeline(0, deposit_df, withdraw_df)
