"""
Download Google BigQuery code from bucket.

  ethereum-block-data
  ethereum-contract-data
  ethereum-token-data
  ethereum-transaction-data

We can download these locally if needed.
"""
import os
import subprocess
from tqdm import tqdm
from typing import Any, List
from google.cloud.storage import Client
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.blob import Blob


def main(args: Any):
    root: str = os.path.join(args.root, args.bucket)
    if not os.path.isdir(root): os.makedirs(root)

    if args.gsutil:
        cmd: str = f'gsutil -m cp -r gs://{args.bucket} {root}'
        subprocess.call(cmd, shell=True)
    else:
        client: Client = Client()
        assert len(os.environ['GOOGLE_APPLICATION_CREDENTIALS']) > 0, \
            "Set GOOGLE_APPLICATION_CREDENTIALS prior to use."

        bucket: Bucket = client.get_bucket(args.bucket)
        blobs: List[Blob] = list(bucket.list_blobs())

        print(f'Found {len(blobs)} in bucket: {args.bucket}.')

        for i in tqdm(range(len(blobs))):
            blob: Blob = blobs[i]
            blob.download_to_filename(os.path.join(root, blob.name))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/bigquery', 
                        help='path to data root (default: ./data/bigquery)')
    parser.add_argument('--bucket', type=str, default='ethereum-block-data',
                        choices=['ethereum-block-data', 
                                 'ethereum-contract-data',
                                 'ethereum-token-data', 
                                 'ethereum-transaction-1week-data', 
                                 'ethereum-transaction-1month-data',
                                 'ethereum-transaction-1year-data',
                                 'ethereum-transaction-data',
                                 'ethereum-transaction-data2'])
    parser.add_argument('--gsutil', action='store_true', default=False)
    args: Any = parser.parse_args()

    main(args)
