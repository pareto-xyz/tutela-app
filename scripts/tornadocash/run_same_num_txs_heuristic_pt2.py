import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from typing import Any, Dict, List, Set
from src.utils.utils import to_json, Entity, Heuristic


def  main(args: Any):
    tx_df: pd.DataFrame = pd.read_csv(
        os.path.join(args.data_dir, 'same_num_txs_transactions.csv'))
    clusters: List[Set[str]] = make_transaction_graph(tx_df)
    clusters_file: str = os.path.join(
        args.data_dir, 'same_num_txs_clusters.json')
    to_json(clusters, clusters_file)
    del tx_df

    tx2addr: pd.DataFrame = pd.read_csv(
        os.path.join(args.data_dir, 'same_num_txs_tx2addr.csv'))
    tx2addr: pd.DataFrame = tx2addr.unique()
    tx2addr: Dict[str, str] = dict(zip(tx2addr.transaction, tx2addr.address))
    tx2addr_file: str = os.path.join(
        args.data_dir, 'same_num_txs_tx2addr.json')
    to_json(tx2addr, tx2addr_file)
    del tx2addr

    addr_df: pd.DataFrame = pd.read_csv(
        os.path.join(args.data_dir, 'same_num_txs_address_sets.csv'))
    addr_df: pd.DataFrame = addr_df.drop_duplicates()
    address_set: Set[str] = make_address_sets(addr_df)
    address_file: str = os.path.join(
        args.data_dir, 'same_num_txs_address_set.json')
    to_json(address_set, address_file)
    del address_set
    del addr_df

    metadata_df: pd.DataFrame = get_metadata(addr_df)
    metadata_df.to_csv(
        os.path.join(args.data_dir, 'same_num_txs_metdata.csv'), index=False)
    del metadata_df


def make_address_sets(address_df: pd.DataFrame) -> List[Set[str]]:
    address_sets: List[Set[str]] = []
    for withdraw_addr, group in address_df.groupby('withdraw_addr'):
        address_sets: List[Set[str]] = \
            address_sets.union({withdraw_addr}, set(group.deposit_addr))
    return address_sets


def make_transaction_graph(transaction_df: pd.DataFrame) -> List[Set[str]]:
    withdraw_tx: np.array = transaction_df.withdraw_tx.to_numpy()
    deposit_tx: np.array = transaction_df.deposit_tx.to_numpy()

    all_tx: np.array = np.unique(np.concatenate([withdraw_tx, deposit_tx]))

    graph: nx.DiGraph = nx.DiGraph()
    graph.add_nodes_from(list(all_tx))
    graph.add_edges_from(list(zip(withdraw_tx, deposit_tx)))

    clusters: List[Set[str]] = [  # ignore singletons
        c for c in nx.weakly_connected_components(graph) if len(c) > 1]

    return clusters


def get_metadata(address_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stores metadata about addresses to add to db. 
    """
    address_df: pd.DataFrame = pd.concat([
        address_df.withdraw_addr, address_df.deposit_addr])
    address_df: pd.DataFrame = address_df.unique()
    addresses: np.array = address_df.to_numpy()
    size: int = addresses.shape[0]

    address: List[str] = addresses.tolist()
    entity: List[int] = (np.ones(size) * Entity.EOA.value).astype(int).tolist()
    conf: List[float] = np.ones(size).astype(int).tolist()
    meta_data: List[str] = [json.dumps({}) for _ in range(len(size))]
    heuristic: List[int] = (np.ones(size) * Heuristic.SAME_NUM_TX.value).astype(int).tolist()

    response: Dict[str, List[Any]] = dict(
        address = address,
        entity = entity,
        conf = conf,
        meta_data = meta_data,
        heuristic = heuristic,
    )
    response: pd.DataFrame = pd.DataFrame.from_dict(response)
    return response


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to tornado cash data')
    args: Any = parser.parse_args()

    main(args)
