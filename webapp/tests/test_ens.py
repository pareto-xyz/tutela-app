"""
Test how well deposit address reuse is doing by computing recall 
of a fixed set of ENS clusters.
"""

import os, json
import pandas as pd
from tqdm import tqdm
from typing import Any, Set, List, Optional

from app.models import Address, Embedding


class TestENSClusters:

    def __init__(self, csv_file: str, mode: str = 'dar'):
        assert mode in ['dar', 'node', 'both'], 'Unexpected mode.'
        self._mode: str = mode
        self._csv_file: str = csv_file
        self._df = pd.read_csv(csv_file)
        self._clusters: List[Set[str]] = self._get_clusters(self._df)

    def _get_clusters(self, df):
        clusters: List[Set[str]] = []
        for _, group in df.groupby('name'):
            cluster: Set[str] = set(group.address)
            if len(cluster) > 1:  # solo clusters are not worthwhile
                clusters.append(cluster)

        return clusters

    def _get_prediction(self, address) -> Set[str]:
        if self.mode == 'dar':
            return self._get_dar_prediction(address)
        elif self.mode == 'node':
            return self._get_node_prediction(address)
        elif self.mode == 'both':
            dar_cluster: Set[str] = self._get_dar_prediction(address)
            node_cluster: Set[str] = self._get_node_prediction(address)
            cluster: Set[str] = set()
            cluster: Set[str] = cluster.union(dar_cluster)
            cluster: Set[str] = cluster.union(node_cluster)
            return cluster

    def _get_dar_prediction(self, address) -> Set[str]:
        addr: Optional[Address] = \
            Address.query.filter_by(address = address).first()

        if addr is not None:
            assert addr.entity == 0, "Address must be an EOA."
            cluster: List[Address] = []
            if (addr.user_cluster is not None) and (addr.user_cluster != -1):
                cluster: List[Address] = Address.query.filter_by(
                    user_cluster = addr.user_cluster).limit(100000).all()
                if cluster is not None:
                    cluster += cluster
            cluster = set([
                c.address for c in cluster if c.entity == 0])  # EOA only
        else:  # if no address, then just return itself
            cluster: Set[str] = {address}

        return cluster

    def _get_node_prediction(self, address) -> Set[str]:
        node: Optional[Embedding] = \
            Embedding.query.filter_by(address = address).first()
        if node is not None:
            # I mapped neighbors -> distances so this is actually loading neighbors
            cluster: List[str] = json.loads(node.distances)
            cluster: Set[str] = set(cluster)
        else:
            cluster: Set[str] = {address}
        return cluster

    def evaluate(self):
        avg_precision: float = 0
        avg_recall: float = 0

        pbar = tqdm(total=len(self._clusters))
        for cluster in self._clusters:
            address: str = list(cluster)[0]  # representative address
            pred_cluster: Set[str] = self._get_prediction(address)

            tp: int = 0  # true positive
            fp: int = 0  # false positive
            for member in pred_cluster:
                member: str = member
                if member in cluster:
                    tp += 1
                else:
                    fp += 1

            fn: int = 0  # false negatives
            for member in cluster:
                member: str = member

                if member not in pred_cluster:
                    fn += 1

            precision: float = get_precision(tp, fp)
            recall: float = get_recall(tp, fn)

            avg_precision += precision
            avg_recall += recall

            pbar.update()

        pbar.close()

        avg_precision /= float(len(self._clusters))
        avg_recall /= float(len(self._clusters))

        return {'precision': avg_precision, 'recall': avg_recall}


def get_precision(tp, fp):
    return tp / float(tp + fp)


def get_recall(tp, fn):
    return tp / float(tp + fn)


if __name__ == "__main__":
    import argparse

    cur_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv-file',
        type=str, 
        default=os.path.join(cur_dir, 'data/ens_pairs.csv'),
        help='path to csv file of ENS clusters',
    )
    parser.add_argument(
        '--max-cluster-size',
        type=int,
        default=100000,
        help='maximum amount of clusters to show',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='dar',
        help='which model to use (dar|node|both)',
        choices=['dar', 'node', 'both'],
    )
    args: Any = parser.parse_args()

    print(TestENSClusters(args.csv_file).evaluate())
