"""
After running `get_neighbors.py` we will dump the information
into a CSV, converting numbers back to address string.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Any, List, Dict

from src.utils.utils import from_json


def main(args: Any):
    distances: np.array = np.load(args.distance_file)
    neighbors: np.array = np.load(args.neighbor_file)
    index2addr: Dict[str, int] = from_json(args.address_file)
    size: int = len(distances)

    address_df: List[str] = []
    distance_df: List[str] = []
    neighbor_df: List[str] = []

    for index in range(size):
        distance: List[float] = distances[index].tolist()
        distance: str = json.dumps(distance)
        neighbor: List[int] = neighbors[index].tolist()
        neighbor: List[str] = [index2addr[nei] for nei in neighbor]
        neighbor: str = json.dumps(neighbor)
        distance_df.append(distance)
        neighbor_df.append(neighbor)
        address_df.append(index2addr[index])

    df_dict: Dict[str, List[str]] = dict(
        address=address_df, 
        distance=distance_df,
        neighbor=neighbor_df,
    )
    df: pd.DataFrame = pd.DataFrame.from_dict(df_dict)
    df.to_csv(os.path.join(args.save_dir, 'diff2vec-processed.csv'), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('distance_file', type=str, help='path to distance numpy file.')
    parser.add_argument('neighbor_file', type=str, help='path to neighbor numpy file.')
    parser.add_argument('address_file', type=str, help='path to address lookup file.')
    parser.add_argument('save_dir', type=str, help='where to save outpouts.')
    args: Any = parser.parse_args()

    main(args)
