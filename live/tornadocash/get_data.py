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
from os.path import join, isdir
from typing import Tuple, Optional, List

from live import utils


def get_last_block():
    """
    Read the current transactions dataframe to see what the latest block is.
    We will grab all data from the block and after.
    """
    data_path:  str = utils.CONSTANTS['data_path']
    cache_path: str = join(data_path, 'live/tornado_cache')
    transactions: pd.DataFrame = pd.read_csv(join(cache_path, 'tornado_transactions.csv'))
    last_block: int = int(transactions.block_number.max())

    return last_block


def update_bigquery(start_block: Optional[int] = None):
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
        ],
        flags = [
            f'--destination_table {project}:tornado_transactions.traces',
            '--use_legacy_sql=false',
        ],
    )
    transaction_query: str = make_bq_query(
        f'select * from {bq_transaction} as b',
        where_clauses = [f'b.hash in ({subtrace_sql})'],
        flags = [
            f'--destination_table {project}:tornado_transactions.transactions',
            '--use_legacy_sql=false',
        ],
    )
    trace_success: bool = utils.execute_bash(trace_query)
    transaction_success: bool = utils.execute_bash(transaction_query)

    return trace_success and transaction_success


def make_bq_query(
    select: str, where_clauses: List[str] = [], flags: List[str] = []) -> str:
    flags: str = ' '.join(flags)
    where_clauses: List[str] = [f'({clause})' for clause in where_clauses]
    where_clauses: str = ' and '.join(where_clauses)
    query: str = f"bq_query {flags} '{select} where {where_clauses}'"
    return query


def get_deposit_and_withdraw(
    trace_df: pd.DataFrame, 
    transaction_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pass


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
    success: bool = update_bigquery(last_block)

    if not success:
        logger.error('failed on updating bigquery tables')
        sys.exit(0)


if __name__ == "__main__":
    main()
