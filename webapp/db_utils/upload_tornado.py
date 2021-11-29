"""
Upload tornado data into PostgreSQL.
"""

import os
import psycopg2
from typing import Any, List


def main(args: Any):
    deposit_csv_path: str = os.path.realpath(args.deposit_csv)
    withdraw_csv_path: str = os.path.realpath(args.withdraw_csv)

    conn = psycopg2.connect(database = 'tornado', user = 'postgres')
    cursor = conn.cursor()

    deposit_columns: List[str] = [
        'hash', 'transaction_index', 'from_address', 'to_address', 'gas',
        'gas_price', 'block_number', 'block_hash', 'tornado_cash_address'
    ]
    cursor.execute(
        f"COPY tornado_deposit({','.join(deposit_columns)}) FROM '{deposit_csv_path}' DELIMITER ',' CSV HEADER;"
    )
    conn.commit()

    withdraw_columns: List[str] = [
        'hash', 'transaction_index', 'from_address', 'to_address', 'gas',
        'gas_price', 'block_number', 'block_hash', 'tornado_cash_address',
        'recipient_address',
    ]
    cursor.execute(
        f"COPY tornado_withdraw({','.join(withdraw_columns)}) FROM '{withdraw_csv_path}' DELIMITER ',' CSV HEADER;"
    )
    conn.commit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('deposit_csv', type=str)
    parser.add_argument('withdraw_csv', type=str)
    args = parser.parse_args()

    main(args)
