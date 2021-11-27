"""
Combine clusters into metadata.csv. To be run after `prune_metadata.csv`.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict
from src.utils.utils import from_json


def main(args: Any):
    df: pd.DataFrame = pd.read_csv(args.metadata_final_csv)

    user_clusters = from_json(args.user_clusters_json)
    exchange_clusters = from_json(args.exchange_clusters_json)

    print('adding user clusters...')
    user_map: Dict[str, int] = {}
    pbar = tqdm(total=len(user_clusters))
    for i, cluster in enumerate(user_clusters):
        for address in cluster:
            user_map[address] = i
        pbar.update()
    pbar.close()

    print('adding exchange clusters...')
    exchange_map: Dict[str, int] = {}
    pbar = tqdm(total=len(exchange_clusters))
    for i, cluster in enumerate(exchange_clusters):
        for address in cluster:
            exchange_map[address] = i
        pbar.update()
    pbar.close()

    df['user_cluster'] = df.address.apply(
        lambda address: user_map.get(address, np.nan),
    )
    df['exchange_cluster'] = df.address.apply(
        lambda address: exchange_map.get(address, np.nan),
    )
    
    # cast to the right type
    df['user_cluster'] = df['user_cluster'].astype(pd.Int64Dtype())
    df['exchange_cluster'] = df['exchange_cluster'].astype(pd.Int64Dtype())

    print('saving to disk...')
    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata_final_csv', type=str)
    parser.add_argument('user_clusters_json', type=str)
    parser.add_argument('exchange_clusters_json', type=str)
    parser.add_argument('out_csv', type=str)
    args = parser.parse_args()

    main(args)
