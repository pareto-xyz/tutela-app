"""
Contains utilities to compute heuristics on all tornado cash data.

1) This assumes get_data.py has been run. It will need access to 
updated files complete_withdraw_tx.csv and complete_deposit_tx.csv. 

2) Run the following heuristics in order:
    - ExactMatch
    - UniqueGasPrice
    - SameNumTx
    - LinkedTx
    - TornMine

3) Run json_to_sql_format functions to generate a processed.csv for 
each heuristic. We will not write any other files to disk.

4) Delete existing content in db. Insert new CSV files into db.
If possible look into ovewriting here rather than deleting rows.
"""
import os
import sys
import pandas as pd
from os.path import join
from typing import List, Any

from live import utils
from src.tcash.heuristic import (
    BaseHeuristic,
    ExactMatchHeuristic,
    GasPriceHeuristic,
    SameNumTransactionsHeuristic,
    LinkedTransactionHeuristic,
    TornMiningHeuristic,
)


def load_input_data():
    data_path:  str = utils.CONSTANTS['data_path']
    out_dir = join(data_path, 'live/tornado_cash')
    trace = pd.read_csv(join(out_dir, 'tornado_traces.csv'))
    transaction = pd.read_csv(join(out_dir, 'tornado_traces.csv'))

    return trace, transaction


def main(args: Any):
    log_path: str = utils.CONSTANTS['log_path']
    os.makedirs(log_path, exist_ok=True)

    log_file: str = join(log_path, 'tornadocash-heuristic.log')
    
    if os.path.isdir(log_file):
        os.remove(log_file)  # remove old file (yesterday's)

    logger = utils.get_logger(log_file)

    data_path: str = utils.CONSTANTS['data_path']
    tx_root: str = join(data_path, 'live/tornado_cash')
    tcash_root: str = join(data_path, 'static/tcash')
    proc_root: str = join(tx_root, 'processed')

    heuristics: List[BaseHeuristic] = [
        # NOTE: these names are the same as the database names.
        ExactMatchHeuristic('exact_match', tx_root, tcash_root, by_pool=True),
        GasPriceHeuristic('gas_price', tx_root, tcash_root, by_pool=True),
        SameNumTransactionsHeuristic('multi_denom', tx_root, tcash_root, max_num_days=1),
        LinkedTransactionHeuristic('linked_transaction',tx_root, tcash_root),
        TornMiningHeuristic('torn_mine', tx_root, tcash_root),
    ]

    for i, heuristic in enumerate(heuristics):
        if args.heuristic >= 0:
            if i != args.heuristic:
                continue

        logger.info(f'entering heuristic {i+1}')

        if args.debug:
            heuristic.run()
        else:
            try:
                heuristic.run()
            except:
                logger.error(f'failed in heuristic {i+1}')
                sys.exit(0)

        name: str = heuristic._name

        if not args.no_db:
            import psycopg2

            save_file: str = join(proc_root, f'{name}.csv')

            conn: Any = psycopg2.connect(
                database = utils.CONSTANTS['postgres_db'], 
                user = utils.CONSTANTS['postgres_user'],
            )
            cursor: Any = conn.cursor()

            cursor.execute(f"DELETE FROM {name}") # delete all rows from table

            columns: List[str] = ['address', 'transaction', 'block_number',
                                  'block_ts', 'meta_data', 'cluster']
            columns: str = ','.join(columns)
            command: str = f"COPY {name}({columns}) FROM '{save_file}' DELIMITER ',' CSV HEADER;"
            cursor.execute(command)  # write CSV to db

            conn.commit()

            cursor.close()
            conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-db', action='store_true', default=False)
    parser.add_argument('--heuristic', type=int, default=-1, help='index of heuristic to run (index: -1)')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
