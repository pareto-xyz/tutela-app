"""
Tools to query for results from Google BigQuery. For context, read
https://evgemedvedev.medium.com/ethereum-blockchain-on-google-bigquery-283fb300f579

The public data is held under `bigquery-public-data.crypto_ethereum`.

Here is a list of all the tables:
    bigquery-public-data.crypto_ethereum.amended_tokens
    bigquery-public-data.crypto_ethereum.balances
    bigquery-public-data.crypto_ethereum.blocks
    bigquery-public-data.crypto_ethereum.contracts
    bigquery-public-data.crypto_ethereum.logs
    bigquery-public-data.crypto_ethereum.token_transfers
    bigquery-public-data.crypto_ethereum.tokens
    bigquery-public-data.crypto_ethereum.traces
    bigquery-public-data.crypto_ethereum.transactions
"""
import os
import pandas as pd
from typing import List, Any, Optional, Tuple, Set
from google.cloud.bigquery import (
    Client, DatasetReference, DestinationFormat, job)
from google.api_core.page_iterator import HTTPIterator
from google.cloud.bigquery.table import RowIterator


class EthereumBigQuery:
    """
    Wrapper class to copy Ethereum blockchain data from BigQuery to buckets.
    """

    def __init__(self, dataset_id: str = 'bigquery-public-data.crypto_ethereum'):
        self.client: Client = Client()
        assert len(os.environ['GOOGLE_APPLICATION_CREDENTIALS']) > 0, \
            "Set GOOGLE_APPLICATION_CREDENTIALS prior to use."
        self.dataset_id: str = dataset_id

    def get_table_names(self) -> List[str]:
        tables: List[HTTPIterator] = self.client.list_tables(self.dataset_id)
        names: List[str] = []
        for table in tables:
            name: str = '{}.{}.{}'.format(
                table.project, table.dataset_id, table.table_id)
            names.append(name)
        return names

    def export_to_bucket(self, table: str, bucket: str, as_json: bool = False):
        """
        Export a full table to a bucket.
        """
        # wildcard notation: 
        # https://cloud.google.com/bigquery/docs/exporting-data#exporting_data_into_one_or_more_files
        # https://github.com/googleapis/python-bigquery/blob/35627d145a41d57768f19d4392ef235928e00f72/docs/snippets.py
        extension: str = 'json' if as_json else 'csv'
        destination_uri: str = f'gs://{bucket}/{table}-*.{extension}'
        project, dataset = self.dataset_id.split('.')
        dataset_ref: DatasetReference = DatasetReference(project, dataset)
        table_ref: DatasetReference = dataset_ref.table(table)

        if as_json:
            job_config = job.ExtractJobConfig()
            job_config.destination_format = DestinationFormat.NEWLINE_DELIMITED_JSON

            extract_job: Any = self.client.extract_table(
                table_ref,
                destination_uri,
                job_config=job_config,
                location='US',
            )
        else:
            extract_job: Any = self.client.extract_table(
                table_ref,
                destination_uri,
                location='US',
            )

        extract_job.result()


if __name__ == "__main__":
    from pprint import pprint

    query: EthereumBigQuery = EthereumBigQuery()
    pprint(query.get_table_names())
