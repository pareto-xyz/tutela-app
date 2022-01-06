import os
from typing import List
from google.cloud.storage import Client
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket


class EthereumStorage:

    def __init__(self):
        self.client: Client = Client()
        assert len(os.environ['GOOGLE_APPLICATION_CREDENTIALS']) > 0, \
            "Set GOOGLE_APPLICATION_CREDENTIALS prior to use."

    def empty_bucket(self, bucket: str):
        bucket: Bucket = self.client.get_bucket(bucket)
        blobs: List[Blob] = bucket.list_blobs()

        for blob in blobs:
            blob.delete()

    def export_to_csv(self, bucket: str, out_dir: str):
        bucket: Bucket = self.client.get_bucket(bucket)
        blobs: List[Blob] = list(bucket.list_blobs())
        for i in range(len(blobs)):
            blob: Blob = blobs[i]
            blob.download_to_filename(os.path.join(out_dir, blob.name))
