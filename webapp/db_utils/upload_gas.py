"""
Upload gas price clusters to  SQL.
"""

import os
import psycopg2
from typing import Any


def main(args: Any):
    csv_path: str = os.path.realpath(args.processed_csv)

    conn = psycopg2.connect(database = 'tornado', user = 'postgres')
    cursor = conn.cursor()

    cursor.execute(
        f"COPY gas_price(address, transaction, block_number, block_ts, meta_data, cluster) FROM '{csv_path}' DELIMITER ',' CSV HEADER;"
    )
    conn.commit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('processed_csv', type=str)
    args = parser.parse_args()

    main(args)
