"""
Pipeline for running a subset of the DAR algorithm

7. Run `heuristic_metadata.py` to generate `metadata-joined.csv`.
8. Run `run_nx.py` to generate `metadata-final.csv`. This is the file that will 
    be used to populate the PostgreSQL database.
9. Run `combine_metadata.py` to add clusters to `metadata-pruned.csv`.

We need to store around a dataframe of the "last chunk" that will be 
contatenated with the current chunk in deposit.py. This last chunk helps
to ensure some degree of scope. We will update this last chunk as we 
process day to day.
"""

import os
import sys
import psycopg2
import numpy as np
import pandas as pd
import networkx as nx
from os.path import join
from typing import List, Any, Tuple, Set

from live import utils
from src.utils.loader import DataframeLoader
from src.cluster.deposit import DepositCluster
from src.utils.utils import from_json

# ---
# begin metadata utilities
# --

def prune_data(df: pd.DataFrame) -> pd.DataFrame:
    exchanges: np.array = df.exchange.unique()

    # exchanges cannot be users or deposits
    df: pd.DataFrame = df[~df.user.isin(exchanges)]
    df: pd.DataFrame = df[~df.deposit.isin(exchanges)]

    # find all users and make sure they cannot be deposits since 
    # deposits cannot send to A -> B -> Exchange.
    users: np.array = df.user.unique()
    df: pd.DataFrame = df[~df.deposit.isin(users)]

    return df


def prune_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = df.loc[df.groupby(['address'])['conf'].idxmax()]
    return df


def merge_metadata(
    dar_metadata: pd.DataFrame,
    tcash_metadata_list: List[pd.DataFrame]) -> pd.DataFrame:

    if 'cluster_type' in dar_metadata.columns:
        dar_metadata.rename(columns={'cluster_type': 'heuristic'}, inplace=True)

    dar_metadata['heuristic'] = 0
    
    if 'metadata' in dar_metadata.columns:
        dar_metadata.rename(columns={'metadata': 'meta_data'}, inplace=True)

    metadata: pd.DataFrame = pd.concat([dar_metadata, *tcash_metadata_list])
    metadata: pd.DataFrame = metadata.loc[
        metadata.groupby('address')['conf'].idxmax()]    

    return metadata

# ---
# begin graph clustering section
# --

def cluster_graph(
    data: pd.DataFrame, 
    tcash_address_list: List[List[Set[str]]],
) -> Tuple[List[Set[str]], List[Set[str]]]:
    user_graph: nx.DiGraph = make_graph(data.user, data.deposit)
    for address_set in tcash_address_list:
        user_graph: nx.DiGraph = add_to_user_graph(user_graph, address_set)

    exchange_graph: nx.DiGraph = make_graph(data.deposit, data.exchange)

    user_wccs: List[Set[str]] = get_wcc(user_graph)
    exchange_wccs: List[Set[str]] = get_wcc(exchange_graph)

    # prune trivial clusters
    user_wccs: List[Set[str]] = remove_singletons(user_wccs)
    exchange_wccs: List[Set[str]] = remove_singletons(exchange_wccs)

    return user_wccs, exchange_wccs


def add_to_user_graph(graph: nx.DiGraph, clusters: List[Set[str]]):
    for cluster in clusters:
        assert len(cluster) == 2, "Only supports edges with two nodes."
        node_a, node_b = cluster
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_edge(node_a, node_b)
    return graph


def get_wcc(graph: nx.DiGraph) -> List[Set[str]]:
    comp_iter: Any = nx.weakly_connected_components(graph)
    comps: List[Set[str]] = [c for c in comp_iter]
    return comps

def remove_deposits(components: List[Set[str]], deposit: Set[str]):
    # remove all deposit addresses from wcc list
    new_components: List[Set[str]] = []
    for component in components:
        new_component: Set[str] = component - deposit
        new_components.append(new_component)

    return new_components


def remove_singletons(components: List[Set[str]]):
    # remove clusters with just one entity... these are not interesting.
    return [c for c in components if len(c) > 1]


def make_graph(node_a: pd.Series, node_b: pd.Series) -> nx.DiGraph:
    """
    DEPRECATED: This assumes we can store all connections in memory.

    Make a directed graph connecting each row of node_a to the 
    corresponding row of node_b.
    """
    assert node_a.size == node_b.size, "Dataframes are uneven sizes."

    graph: nx.DiGraph = nx.DiGraph()

    nodes: np.array = np.concatenate([node_a.unique(), node_b.unique()])
    edges: List[Tuple[str, str]] = list(
        zip(node_a.to_numpy(), node_b.to_numpy())
    )

    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph

# ---
# begin main function
# --

def main(args: Any):
    log_path: str = utils.CONSTANTS['log_path']
    os.makedirs(log_path, exist_ok=True)

    log_file: str = join(log_path, 'depositreuse-heuristic.log')
    os.remove(log_file)  # remove old file (yesterday's)

    logger = utils.get_logger(log_file)

    data_path: str = utils.CONSTANTS['data_path']
    static_path: str = utils.CONSTANTS['static_path']
    depo_path: str = join(data_path, '/live/depositreuse')

    # we will need to put some of these files into the Address db table
    tcash_root: str = join(data_path, 'static/tcash/processed')
    proc_path: str = join(depo_path, 'processed')

    logger.info('loading dataframe for DAR')
    loader: DataframeLoader = DataframeLoader(
        join(depo_path, 'ethereum_blocks_live.csv'),
        join(static_path, 'known_addresses.csv'),
        join(depo_path, 'ethereum_transactions_live.csv'),
        proc_path,
    )
    logger.info('initializing DAR instance')
    heuristic: DepositCluster = DepositCluster(
        loader, a_max = 0.01, t_max = 3200, save_dir = proc_path)

    if args.debug:
        heuristic.make_clusters()
    else:
        try:
            heuristic.make_clusters()
        except:
            logger.error('failed in make_clusters()')
            sys.exit(0)

    data_file: str = join(proc_path, 'data.csv')
    metadata_file: str = join(proc_path, 'metadata.csv')
    tx_file: str = join(proc_path, 'transactions.csv')

    # should be okay for memory
    data: pd.DataFrame = pd.read_csv(data_file)
    metadata: pd.DataFrame = pd.read_csv(metadata_file)

    logger.info('pruning data')
    try:
        data = prune_data(data)
    except:
        logger.error('failed in prune_data()')
        sys.exit(0)

    logger.info('pruning metadata')
    try:
        metadata = prune_data(metadata)
    except:
        logger.error('failed in prune_metadata()')
        sys.exit(0)

    logger.info('loading metadata from tcash heuristics')
    tcash_names: List[str] = [
        'exact_match',
        'gas_price',
        'multi_denom',
        'linked_transaction',
        'torn_mine',
    ]
    tcash_metadata_list: List[pd.DataFrame] = []
    for name in tcash_names:
        df: pd.DataFrame = pd.read_csv(
            join(tcash_root, f'{name}_metadata_live.csv'))
        tcash_metadata_list.append(df)

    logger.info('merging metadata')
    try:
        metadata: pd.DataFrame = merge_metadata(metadata, tcash_metadata_list)
    except:
        logger.error('failed in merge_metadata()')
        sys.exit(0)

    logger.info('loading addresses from tcash heuristics')
    tcash_names: List[str] = [
        'exact_match',
        'gas_price',
        'multi_denom',
        'linked_transaction',
        'torn_mine',
    ]
    tcash_address_list: List[pd.DataFrame] = []
    for name in tcash_names:
        address_set: List[Set[str]] = from_json(
            join(tcash_root, f'{name}_address_live.json'))
        tcash_address_list.append(address_set)

    logger.info('clustering with networkx')
    try:
        user_clusters, exchange_clusters = cluster_graph(
            metadata, tcash_address_list)
    except:
        logger.error('failed in cluster_graph()')
        sys.exit(0)

    # TODO: need to remap these clusters to old ones. this can be tricky.

    # -- algorithm completed at this point: we need to now populate db
    if not args.no_db:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-db', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
