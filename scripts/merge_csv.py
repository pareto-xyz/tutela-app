import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import List, Any


def main(args: Any):
    processed_dir: str = os.path.join(args.root, 'processed')
    if not os.path.isdir(processed_dir): os.makedirs(processed_dir)
    merge_file: str = os.path.join(processed_dir, args.merge_name)

    if os.path.isfile(merge_file):
        print('Merge file already exists. Remove before running.')
        return

    paths: List[str] = sorted(glob(os.path.join(args.root, '*.csv')))

    print(f'Creating merged file: {args.merge_name}.')

    for i in tqdm(range(len(paths))):
        path: str = paths[i]
        df = pd.read_csv(path)
        df.to_csv(merge_file, index=False, header=(i==0), mode='a')
        del df

    if not args.no_sort:
        # I ran this part manually
        df: pd.DataFrame = pd.read_csv(merge_file)
        df: pd.DataFramef = df.sort_values(by=args.sort_column)
        out_name: str = os.path.join(
            args.root,
            'processed',
            args.merge_name.replace('merged', 'sorted'),
        )
        df.to_csv(out_name, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
        default='./data/bigquery/ethereum-block-data', 
        help='path to data root (default: ./data/bigquery/ethereum-block-data)',
    )
    parser.add_argument(
        '--merge-name',
        type=str,
        default='blocks-merged.csv',
        help='name of merged file (default: blocks-merged.csv)',
    )
    parser.add_argument(
        '--sort-column',
        type=str,
        default='number',
        help='name of column to sort by (default: number)',
    )
    parser.add_argument('--no-sort', action='store_true', default=False)
    args: Any = parser.parse_args()

    main(args)
