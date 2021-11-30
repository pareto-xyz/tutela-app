"""
Downloaded known addresses from https://www.kaggle.com/hamishhall/labelled-ethereum-addresses/version/1.
We need to combine this with `exchange.csv` and create a dump of `contract.csv`.
"""
import numpy as np
import pandas as pd
from typing import Any, List


def main(args: Any):
    kaggle_df: pd.DataFrame = pd.read_csv(args.kaggle_csv)

    # spend some time just cleaning the kaggle DF
    keys: List[str] = [
        'Address', 'Name', 'Account Type',
        'Entity', 'Label', 'Tags',
    ]
    kaggle_df: pd.DataFrame = kaggle_df[keys]
    kaggle_df.rename(
        columns={
            'Address': 'address',
            'Name': 'name',
            'Account Type': 'account_type',
            'Entity': 'entity',
            'Label': 'label',
            'Tags': 'tags',
        },
        inplace = True,
    )
    label: pd.Series = kaggle_df.label
    label: pd.Series = label.replace('Legit', 1)
    label: pd.Series = label.replace('Dodgy', 0)
    label: pd.Series = label.astype(bool)
    kaggle_df.label = label

    account_type: pd.Series = kaggle_df.account_type
    account_type: pd.Series = account_type.replace('Smart Contract', 'contract')
    account_type: pd.Series = account_type.replace('Wallet', 'eoa')
    kaggle_df.account_type = account_type

    kaggle_df.entity = kaggle_df.entity.str.lower()
    kaggle_df.tags = kaggle_df.tags.str.lower()

    kaggle_df = kaggle_df.drop_duplicates('address')
    kaggle_df = kaggle_df[~(kaggle_df.address == 'Address')]

    # process etherclust.csv
    etherclust_df: pd.DataFrame = pd.read_csv(args.etherclust_csv)
    etherclust_df: pd.DataFrame = etherclust_df[
        ~etherclust_df.address.isin(kaggle_df.address)]

    etherclust_df.rename(columns={'type': 'entity'}, inplace=True)
    etherclust_df.entity = etherclust_df.entity.str.lower()
    etherclust_df.loc[etherclust_df['entity'] == 'wallet', 'entity'] = np.nan
    etherclust_df['label'] = 1
    etherclust_df['tags'] = np.nan

    combined_df: pd.DataFrame = pd.concat(
        [kaggle_df, etherclust_df],
        ignore_index=True,
    )

    # process etherscan.csv
    etherscan_df: pd.DataFrame = pd.read_csv(args.etherscan_csv)
    etherscan_df.rename(columns={'labels': 'label'}, inplace=True)
    etherscan_df: pd.DataFrame = etherscan_df[
        ~etherscan_df.address.isin(combined_df.address)]
    etherscan_df.entity = etherscan_df.entity.str.lower()
    etherscan_df['tags'] = np.nan

    combined_df: pd.DataFrame = pd.concat(
        [combined_df, etherscan_df],
        ignore_index=True,
    )

    # process tornado.csv
    tornado_df: pd.DataFrame =pd.read_csv(args.tornado_csv)
    tornado_df.rename(columns={
        'legitimacy': 'label',
        'type': 'entity',
    }, inplace=True)
    tornado_df: pd.DataFrame = tornado_df[
        ~tornado_df.address.isin(combined_df.address)]
    tornado_df.entity = tornado_df.entity.str.lower()

    df: pd.DataFrame = pd.concat(
        [combined_df, tornado_df],
        ignore_index=True,
    )
    df.rename(columns={'label': 'legitimacy'}, inplace=True)
    df.to_csv(args.known_csv, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--kaggle-csv',
        type=str,
        default='./data/static/kaggle.csv', 
        help='path to data root (default: ./data/static/kaggle.csv)',
    )
    parser.add_argument(
        '--etherclust-csv',
        type=str,
        default='./data/static/etherclust.csv', 
        help='path to data root (default: ./data/static/etherclust.csv)',
    )
    parser.add_argument(
        '--etherscan-csv',
        type=str,
        default='./data/static/etherscan.csv', 
        help='path to data root (default: ./data/static/etherscan.csv)',
    )
    parser.add_argument(
        '--tornado-csv',
        type=str,
        default='./data/static/tornado.csv', 
        help='path to data root (default: ./data/static/tornado.csv)',
    )
    parser.add_argument(
        '--known-csv',
        type=str,
        default='./data/static/known_addresses.csv', 
        help='path to data root (default: ./data/static/known_addresses.csv)',
    )
    args: Any = parser.parse_args()

    main(args)
