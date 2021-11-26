"""
Lambda Class's Multi-Denomiation Heuristic.
"""
import os
import pandas as pd
from tqdm import tqdm
import networkx as nx
from typing import Any, Tuple, List, Set, Optional, Dict
from src.utils.utils import to_json

TORNADO_ADDRESSES: Dict[set, set] = {
    '0xd4b88df4d29f5cedd6857912842cff3b20c8cfa3': '100 DAI',
    '0xfd8610d20aa15b7b2e3be39b396a1bc3516c7144': '1000 DAI',
    '0x07687e702b410fa43f4cb4af7fa097918ffd2730': '10000 DAI',
    '0x23773e65ed146a459791799d01336db287f25334': '100000 DAI',
    '0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc': '0.1 ETH',
    '0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936': '1 ETH',
    '0x910cbd523d972eb0a6f4cae4618ad62622b39dbf': '10 ETH',
    '0xa160cdab225685da1d56aa342ad8841c3b53f291': '100 ETH',
    '0xd96f2b1c14db8458374d9aca76e26c3d18364307': '100 USDC',
    '0x4736dcf1b7a3d580672cce6e7c65cd5cc9cfba9d': '1000 USDC',
    '0x169ad27a470d064dede56a2d3ff727986b15d52b': '100 USDT',
    '0x0836222f2b2b24a3f36f98668ed8f0b38d1a872f': '1000 USDT',
    '0x178169b423a011fff22b9e3f3abea13414ddd0f1': '0.1 WBTC',
    '0x610b717796ad172b316836ac95a2ffad065ceab4': '1 WBTC',
    '0xbb93e510bbcd0b7beb5a853875f9ec60275cf498': '10 WBTC',
    '0x22aaa7720ddd5388a3c0a3333430953c68f1849b': '5000 cDAI',
    '0x03893a7c7463ae47d46bc7f091665f1893656003': '50000 cDAI',
    '0x2717c5e28cf931547b621a5dddb772ab6a35b701': '500000 cDAI',
    '0xd21be7248e0197ee08e0c20d4a96debdac3d20af': '5000000 cDAI'
}


def main(args: Any):
    withdraw_df, deposit_df = load_data(args.data_dir)
    clusters, tx2addr = get_multi_denomination(deposit_df, withdraw_df)
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    clusters_file: str = os.path.join(args.save_dir, f'multi_denom_clusters.json')
    tx2addr_file: str = os.path.join(args.save_dir, f'multi_denom_tx2addr.json')
    to_json(clusters, clusters_file)
    to_json(tx2addr, tx2addr_file)


def load_data(root) -> Tuple[pd.DataFrame, pd.DataFrame]:
    withdraw_df: pd.DataFrame = pd.read_csv(
        os.path.join(root, 'lighter_complete_withdraw_txs.csv'))

    # Change recipient_address to lowercase.
    withdraw_df['recipient_address'] = withdraw_df['recipient_address'].str.lower()
    
    # Change block_timestamp field to be a timestamp object.
    withdraw_df['block_timestamp'] = withdraw_df['block_timestamp'].apply(pd.Timestamp)

    deposit_df: pd.DataFrame = pd.read_csv(
        os.path.join(root, 'lighter_complete_deposit_txs.csv'))
    
    # Change block_timestamp field to be a timestamp object.
    deposit_df['block_timestamp'] = deposit_df['block_timestamp'].apply(pd.Timestamp)

    return withdraw_df, deposit_df


def get_multi_denomination(
    deposit_df: pd.DataFrame, 
    withdraw_df: pd.DataFrame, 
) -> Tuple[List[Set[str]], Dict[str, str]]:
    """
    If there are multiple (say 12) deposit transactions coming from 
    a deposit address and later there are 12 withdraw transactions 
    to the same withdraw address, then we can link all these deposit 
    transactions to the withdraw transactions.

    In particular, given a withdrawal transaction, an anonymity score 
    is assigned to it:

        1) The number of previous withdrawal transactions with the same 
        address as the given withdrawal transaction is registered.

        2) The deposit transactions data are grouped by their address. 
        Addresses that deposited the same number of times as the number 
        of withdraws registered, are grouped in a set $C$.

        3) An anonymity score (of this heuristic) is assigned to the 
        withdrawal transaction following the formula $P = 1 - 1/|C|$, 
        where P is the anonymity score and $|C|$ is the cardinality 
        of set $C$.
    """
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to tornado cash data')
    parser.add_argument('save_dir', type=str, help='folder to save matches')
    args: Any = parser.parse_args()

    main(args)

