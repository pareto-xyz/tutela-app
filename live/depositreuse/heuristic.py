"""
Pipeline for running a subset of the DAR algorithm.

We need to store around a dataframe of the "last chunk" that will be 
concatenated with the current chunk in deposit.py. This last chunk helps
to ensure some degree of scope. We will update this last chunk as we 
process day to day.

This script will assume access to outputs the tornadocash pipeline. In
particular, we do this to supplement the DAR clusters with tornadocash
address clusters. We optimize for programming ease and not efficiency:
this script will wipe all non-rows from the db and re-populate each run,
even though most of the rows will remain identical.
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from os.path import join
from collections import Counter
from typing import List, Any, Tuple, Set, Dict

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


def add_clusters_to_metadata(
    metadata: pd.DataFrame,
    user_clusters: List[Set[str]],
    exchange_clusters: List[Set[str]],
) -> pd.DataFrame:
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

    metadata['user_cluster'] = metadata.address.apply(
        lambda address: user_map.get(address, np.nan))
    metadata['exchange_cluster'] = metadata.address.apply(
        lambda address: exchange_map.get(address, np.nan))
    
    # cast to the right type
    metadata['user_cluster'] = metadata['user_cluster'].astype(pd.Int64Dtype())
    metadata['exchange_cluster'] = metadata['exchange_cluster'].astype(pd.Int64Dtype())

    return metadata


def merge_clusters_with_db(metadata: pd.DataFrame, greedy: bool = False) -> pd.DataFrame:
    """
    The current user clusters and exchange clusters are enumerated 
    from 0 onward. A simple solution would be to just start from the 
    current maximum M number of clusters (M onward). However, this is 
    not perfect as now an address may be in multiple clusters. We need
    to also do cluster aggregation. Consider the following:

    A cluster with addresses  {`a`, `b`, `c`}
    
    but `a` is already in cluster 0 and `b` is already in cluster 1. 
    Suppose `c` is not in any cluster, and that there are 100 clusters 
    currently. 

    What we do: 

    Create a new cluster 101, that contains all members in cluster 0, 
    cluster 1, and also `c`.
    """
    import psycopg2

    conn: Any = psycopg2.connect(
        database = utils.CONSTANTS['postgres_db'], 
        user = utils.CONSTANTS['postgres_user'])
    cursor: Any = conn.cursor()

    def __merge_clusters(
        metadata: pd.DataFrame, 
        field_name: str = 'user_cluster') -> pd.DataFrame:

        # don't update all the time!
        metadata: pd.DataFrame = metadata[metadata.conf >= 0.5]
        unique_clusters: List[int] = range(
            metadata[field_name][~pd.isna(metadata[field_name])].min(),
            metadata[field_name][~pd.isna(metadata[field_name])].max()+1)
        
        print(f'merging clusters: {field_name}')
        pbar = tqdm(total=len(metadata[field_name]))
        for cluster_ in unique_clusters:
            # for each new cluster found, check if it's already in the db and if so
            # grab the already assigned clusters.
            cluster: pd.DataFrame = metadata[metadata[field_name] == cluster_]
            cluster_size: int = len(cluster)
            # Note that this logic ignores new addresses. New addresses will be 
            # inserted via metadata object below.
            addresses: List[str] = list(cluster.address.unique())

            batch_size: int = 1000  # compute this in batches
            num_batches: int = len(addresses) // batch_size
            old_clusters: List[int] = []
            count: int = 0

            for b in range(num_batches):
                batch: List[str] = addresses[b*batch_size:(b+1)*batch_size]
                batch: List[str] = ["'" + address + "'" for address in batch]
                batch: str = '(' + ','.join(batch) + ')'
                command: str = f"select {field_name} from address where address in {batch}"
                cursor.execute(command)
                out: List[Tuple[Any]] = cursor.fetchall()
                out: List[int] = [x[0] for x in out if x[0] is not None]
                old_clusters.extend(out)
                count += batch_size

            if len(addresses) % batch_size != 0:
                batch: List[str] = addresses[count:]
                batch: List[str] = ["'" + address + "'" for address in batch]
                batch: str = '(' + ','.join(batch) + ')'
                command: str = f"select {field_name} from address where address in {batch}"
                cursor.execute(command)
                out: List[Tuple[Any]] = cursor.fetchall()
                out: List[int] = [x[0] for x in out if x[0] is not None]
                old_clusters.extend(out)

            # `old_clusters` stores all clusters. Find the most common one!
            if len(old_clusters) > 0:
                mode_cluster: int = Counter(old_clusters).most_common()[0][0]
                unique_old_clusters: List[int] = list(set(old_clusters) - set([mode_cluster]))
            else:
                unique_old_clusters: List[int] = []

            if len(unique_old_clusters) > 0 and not args.greedy:
                # if our cluster joins multiple old clusters, we need to make this consistent
                # by merging old clusters. Need to do this in batches too.
                num_batches: int = len(unique_old_clusters) // batch_size
                count: int = 0
                for b in range(num_batches):
                    batch: List[int] = unique_old_clusters[b*batch_size:(b+1)*batch_size]
                    batch: List[str] = [str(x) for x in batch]
                    batch: str = '(' + ','.join(batch) + ')'
                    command: str = f"update address set {field_name} = {mode_cluster} where {field_name} in {batch}"
                    cursor.execute(command)
                    conn.commit()
                    count += batch_size

                if len(unique_old_clusters) % batch_size != 0:
                    batch: List[str] = unique_old_clusters[count:]
                    batch: List[str] = [str(x) for x in batch]
                    batch: str = '(' + ','.join(batch) + ')'
                    command: str = f"update address set {field_name} = {mode_cluster} where {field_name} in {batch}"
                    cursor.execute(command)
                    conn.commit()
                    count += batch_size

            # replace new cluster w/ matched old cluster!
            metadata.loc[metadata[field_name] == cluster_, field_name] = mode_cluster
            pbar.update(cluster_size)
        pbar.close()

        cursor.close()
        conn.close()

        return metadata

    metadata: pd.DataFrame = __merge_clusters(metadata, 'user_cluster')
    metadata: pd.DataFrame = __merge_clusters(metadata, 'exchange_cluster')

    return metadata

# ---
# begin main function
# --

def main(args: Any):
    log_path: str = utils.CONSTANTS['log_path']
    os.makedirs(log_path, exist_ok=True)

    log_file: str = join(log_path, 'depositreuse-heuristic.log')
    if os.path.isfile(log_file):
        os.remove(log_file)  # remove old file (yesterday's)

    logger = utils.get_logger(log_file)

    if not args.db_only:
        data_path: str = utils.CONSTANTS['data_path']
        static_path: str = utils.CONSTANTS['static_path']
        depo_path: str = join(data_path, 'live/depositreuse')

        # we will need to put some of these files into the Address db table
        tcash_root: str = join(data_path, 'live/tornado_cash/processed')
        proc_path: str = join(depo_path, 'processed')

        if not args.no_algo:
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

            # set last chunk
            logger.info('loading last-chunk')
            lastchunk_path: str = join(proc_path, 'transactions-lastchunk.csv')
            lastchunk: pd.DataFrame = pd.read_csv(lastchunk_path)
            heuristic.set_last_chunk(lastchunk)

            if args.debug:
                heuristic.make_clusters()
            else:
                try:
                    heuristic.make_clusters()
                except:
                    logger.error('failed in make_clusters()')
                    sys.exit(0)

            # get new last chunk
            logger.info('saving new last-chunk')
            lastchunk: pd.DataFrame = heuristic.get_last_chunk()
            lastchunk.to_csv(lastchunk_path, index=False)
        
        # --
        data_file: str = join(proc_path, 'data.csv')
        metadata_file: str = join(proc_path, 'metadata.csv')
        tx_file: str = join(proc_path, 'transactions.csv')

        # should be okay for memory
        data: pd.DataFrame = pd.read_csv(data_file)
        metadata: pd.DataFrame = pd.read_csv(metadata_file)

        logger.info('pruning data')
        if args.debug:
            data = prune_data(data)
        else:
            try:
                data = prune_data(data)
            except:
                logger.error('failed in prune_data()')
                sys.exit(0)

        logger.info('pruning metadata')
        if args.debug:
            metadata = prune_metadata(metadata)
        else:
            try:
                metadata = prune_metadata(metadata)
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
            df: pd.DataFrame = pd.read_csv(join(tcash_root, f'{name}_metadata.csv'))
            tcash_metadata_list.append(df)

        logger.info('merging metadata')
        if args.debug:
            metadata: pd.DataFrame = merge_metadata(metadata, tcash_metadata_list)
        else:
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
            # only load if it exists
            address_file: str  = join(tcash_root, f'{name}_address.json')
            if os.path.isfile(address_file):
                address_set: List[Set[str]] = from_json(address_file)
                tcash_address_list.append(address_set)

        logger.info('clustering with networkx')
        if args.debug:
            user_clusters, exchange_clusters = cluster_graph(
                data, tcash_address_list)
        else:
            try:
                user_clusters, exchange_clusters = cluster_graph(
                    data, tcash_address_list)
            except:
                logger.error('failed in cluster_graph()')
                sys.exit(0)

        logger.info('adding clusters into metadata')
        if args.debug:
            metadata: pd.DataFrame = add_clusters_to_metadata(
                metadata, user_clusters, exchange_clusters)
        else:
            try:
                metadata: pd.DataFrame = add_clusters_to_metadata(
                    metadata, user_clusters, exchange_clusters)
            except:
                logger.error('failed in add_clusters_to_metadata()')
                sys.exit(0)

        # save new metadata to file
        metadata_file: str = join(proc_path, 'metadata.csv')
        metadata.to_csv(metadata_file, index=False)
    else:
        data_path: str = utils.CONSTANTS['data_path']
        depo_path: str = join(data_path, 'live/depositreuse')
        proc_path: str = join(depo_path, 'processed')
        metadata_file: str = join(proc_path, 'metadata.csv')
        tx_file: str = join(proc_path, 'transactions.csv')

    # -- 
    # algorithm completed at this point: we need to now populate db
    if not args.no_db:
        metadata: pd.DataFrame = pd.read_csv(metadata_file)

        # need to recast upon loading csv
        metadata.user_cluster = metadata.user_cluster.astype(pd.Int64Dtype())
        metadata.exchange_cluster = metadata.exchange_cluster.astype(pd.Int64Dtype())

        # merge these user_clusters consistently with the existing
        # clusters such that any address is in only one address
        logger.info('merging clusters in current metadata with db')
        if args.debug:
            metadata: pd.DataFrame = merge_clusters_with_db(metadata, greedy=args.greedy)
        else:
            try:
                metadata: pd.DataFrame = merge_clusters_with_db(metadata, greedy=args.greedy)
            except:
                logger.error('failed in merge_clusters_with_db()')
                sys.exit(0)

        merged_file: str = join(proc_path, 'metadata-merged.csv')
        metadata.to_csv(merged_file, index=False)

        import psycopg2

        conn: Any = psycopg2.connect(
            database = utils.CONSTANTS['postgres_db'], 
            user = utils.CONSTANTS['postgres_user'])
        cursor: Any = conn.cursor()

        # step 1: delete rows from address staging table where TCash
        cursor.execute("delete from address_staging");
        conn.commit()

        # step 2: insert metadata rows into address table: this includes 
        # all TCash and includes most recent DAR
        columns: List[str] = [
            'address',
            'entity',
            'conf',
            'meta_data',
            'heuristic',
            'user_cluster',
            'exchange_cluster',
        ]
        columns: str = ','.join(columns)
        # copy to an empty table `address_staging`
        command: str = f"COPY address_staging({columns}) FROM '{merged_file}' DELIMITER ',' CSV HEADER;"
        cursor.execute(command)
        conn.commit()

        # step 3: insert values from `address_staging` into `address`.
        set_sql: str = f"set user_cluster = EXCLUDED.user_cluster, exchange_cluster = EXCLUDED.exchange_cluster"
        command: str = f"insert into address({columns}) select {columns} from address_staging on conflict (address) do update {set_sql};"
        cursor.execute(command)
        conn.commit()

        # step 4: insert transactions into deposit_transactions table
        columns: List[str] = [
            'address',
            'deposit',
            'transaction',
            'block_number',
            'block_ts',
            'conf',
        ]
        columns: str = ','.join(columns)
        cursor.execute(f"COPY deposit_transaction({columns}) FROM '{tx_file}' DELIMITER ',' CSV HEADER;")
        cursor.execute(command)
        conn.commit()

        cursor.close()
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-db', action='store_true', default=False,
                        help='skip the code to edit database (default: False)')
    parser.add_argument('--no-algo', action='store_true', default=False,
                        help='skip the DAR algorithm but do everything else (default: False)')
    parser.add_argument('--db-only', action='store_true', default=False,
                        help='only execute the code to edit database (default: False)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='throw errors / no try-catch (default: False)')
    parser.add_argument('--greedy', action='store_true', default=False,
                        help='do not correct old clusters, just new ones (default: False)')
    args = parser.parse_args()

    main(args)
