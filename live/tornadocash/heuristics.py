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
from typing import Tuple, Optional, List, Dict, Any

from live import utils


def load_input_data():
    data_path:  str = utils.CONSTANTS['data_path']
    out_dir = join(data_path, 'live/tornado_cash')
    trace = pd.read_csv(join(out_dir, 'tornado_traces.csv'))
    transaction = pd.read_csv(join(out_dir, 'tornado_traces.csv'))

    return trace, transaction


def main():
    log_path: str = utils.CONSTANTS['log_path']
    os.makedirs(log_path, exist_ok=True)

    log_file: str = join(log_path, 'tornadocash-data.log')
    os.remove(log_file)  # remove old file (yesterday's)

    logger = utils.get_logger(log_file)

    logger.info('loading deposit/withdraw dataframes')
    trace, transaction = load_input_data()


if __name__ == "__main__":
    main()
