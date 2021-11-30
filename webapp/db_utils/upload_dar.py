"""
For large amounts of data, it is too slow to use FlaskSQL. 
Try to upload directly using psycopg2 a large CSV file!

Make sure to run `combine_metadata.py` and `prune_metadata.py`
before running this.
"""

import os
import psycopg2
from typing import Any


def main(args: Any):
    csv_path: str = os.path.realpath(args.metadata_csv)

    conn = psycopg2.connect(database = 'tornado', user = 'postgres')
    cursor = conn.cursor()

    cursor.execute(
        f"COPY address(address, entity, conf, meta_data, heuristic, user_cluster, exchange_cluster) FROM '{csv_path}' DELIMITER ',' CSV HEADER;"
    )
    conn.commit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata_csv', type=str)
    args = parser.parse_args()

    main(args)
