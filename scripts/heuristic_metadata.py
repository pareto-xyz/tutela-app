"""
We made separate metadata for heuristics and need to add it the 
`metadata-pruned.csv` file. We will save it as `metadata-joined.csv`
"""

import pandas as pd
from tqdm import tqdm
from typing import Any


def main(args: Any):
    dar_metadata: pd.DataFrame = pd.read_csv(args.metadata_pruned)
    if 'cluster_type' in dar_metadata.columns:
        dar_metadata.rename(columns={'cluster_type': 'heuristic'}, inplace=True)
    dar_metadata['heuristic'] = 0
    if 'metadata' in dar_metadata.columns:
        dar_metadata.rename(columns={'metadata': 'meta_data'}, inplace=True)
    gas_metadata: pd.DataFrame = pd.read_csv(args.gas_price_metadata)
    same_metadata: pd.DataFrame = pd.read_csv(args.same_num_tx_metadata)

    metadata: pd.DataFrame = pd.concat([dar_metadata, gas_metadata, same_metadata])
    metadata: pd.DataFrame = metadata.loc[metadata.groupby('address')['conf'].idxmax()]
    metadata.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata_pruned', type=str)
    parser.add_argument('gas_price_metadata', type=str)
    parser.add_argument('same_num_tx_metadata', type=str)
    parser.add_argument('out_csv', type=str)
    args = parser.parse_args()

    main(args)
