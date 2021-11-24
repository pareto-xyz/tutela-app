import pickle
from typing import Dict, List, Optional

from src.utils.loader import DataLoader


class BaseCluster:
    """
    Inherit me for clustering algorithms.
    """

    def __init__(self, loader: DataLoader):
        self.loader: DataLoader = loader

    def make_clusters(self) -> Optional[Dict[str, List[str]]]:
        """
        Main function to overwrite. Assigns clusters to addresses.
        """
        return dict()  # does nothing

    def _save(self, x, file):
        with open(file, 'wb') as fp:
            pickle.dump(x, fp, pickle.HIGHEST_PROTOCOL)

    def _load(self, file):
        with open(file, 'rb') as fp:
            return pickle.load(fp)
