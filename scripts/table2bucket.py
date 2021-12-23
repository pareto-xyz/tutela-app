from typing import Any
from src.utils.bigquery import EthereumBigQuery


def main(args: Any):
    query: EthereumBigQuery = EthereumBigQuery(args.project)
    # pprint(query.get_table_names())
    query.export_to_bucket(args.bucket, args.table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='bigquery-public-data.crypto_ethereum',
                        choices=['lexical-theory-329617.crypto_ethereum', 
                                 'bigquery-public-data.crypto_ethereum'])
    parser.add_argument('--bucket', type=str, default='blocks',
                        choices=['blocks', 'transactions', 'transactions2', 'tokens', 'token_transfers', 
                                 'traces', 'contracts', 'transactions_1week',
                                 'transactions_1month', 'transactions_1year'])
    parser.add_argument('--table', type=str, default='ethereum-block-data',
                        choices=['ethereum-block-data', 'ethereum-transaction-data',
                                 'ethereum-transaction-data2',
                                 'ethereum-token-data', 'ethereum-transfers-data', 
                                 'ethereum-trace-data', 'ethereum-contract-data',
                                 'ethereum-transaction-1week-data',
                                 'ethereum-transaction-1month-data',
                                 'ethereum-transaction-1year-data'])
    args: Any = parser.parse_args()

    main(args)
