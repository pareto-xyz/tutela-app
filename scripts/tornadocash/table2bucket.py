from typing import Any
from src.utils.bigquery import EthereumBigQuery


def main(args: Any):
    query: EthereumBigQuery = EthereumBigQuery(args.project)
    query.export_to_bucket(args.bucket, args.table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='lexical-theory-329617.tornado_transactions')
    parser.add_argument('--bucket', type=str, default='traces',
                        choices=['traces', 'transactions', 'tornadocontracts'])
    parser.add_argument('--table', type=str, default='tornado-trace',
                        choices=['tornado-trace', 'tornado-transaction', 'tornado-contract'])
    args: Any = parser.parse_args()

    main(args)
