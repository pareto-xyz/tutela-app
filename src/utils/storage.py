import os
from typing import List, Any
from google.cloud.storage import Client, Blob


class EthereumStorage:

    def __init__(self):
        self.client: Client = Client()
        assert len(os.environ['GOOGLE_APPLICATION_CREDENTIALS']) > 0, \
            "Set GOOGLE_APPLICATION_CREDENTIALS prior to use."

    def empty_bucket(self, bucket: str):
        bucket: Any = self.client.bucket(bucket)
        blobs: List[Blob] = bucket.list_blobs()

        for blob in blobs:
            blob.delete()
