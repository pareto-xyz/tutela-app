"""
Lambda Class's Exact Match Heuristic.
"""
import os
import pandas as pd
from tqdm import tqdm
import networkx as nx
from typing import Any, Tuple, List, Set, Optional, Dict
from src.utils.utils import to_json


def main(args: Any):
    withdraw_df, deposit_df = load_data(args.data_dir)
    clusters, tx2addr = get_exact_matches(deposit_df, withdraw_df, by_pool=args.by_pool)
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    appendix: str = '_by_pool' if args.by_pool else ''
    # NOTE: we do not make a `db_file` here b/c we are guaranteed singleton clusters.
    clusters_file: str = os.path.join(
        args.save_dir, f'exact_match_clusters{appendix}.json')
    tx2addr_file: str = os.path.join(
        args.save_dir, f'exact_match_tx2addr{appendix}.json')
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


def get_exact_matches(
    deposit_df: pd.DataFrame, 
    withdraw_df: pd.DataFrame, 
    by_pool: bool = False,
) -> Tuple[List[Set[str]], Dict[str, str], Dict[str, Any]]:
    """
    Iterate over the withdraw transactions and apply heuristic one. For each 
    withdraw with matching deposit transactions, a new element is added to 
    the dictionary with the key as the withdraw hash and the values as all 
    matching deposit hashes.

    It is possible for a transaction hash to appear more than once. As such, 
    we compute weakly connected components to form clusters.
    """
    tx2addr: Dict[str, str] = {}
    graph: nx.DiGraph = nx.DiGraph()
    raw_links: Dict[str, str] = {}

    for withdraw_row in tqdm(withdraw_df.itertuples(), total=withdraw_df.shape[0]):
        heuristic_fn = \
            exact_match_heuristic_by_pool if by_pool else exact_match_heuristic
        results: Tuple[bool, List[pd.Series]] = \
            heuristic_fn(deposit_df, withdraw_row)

        if results[0]:
            deposit_rows: List[pd.Series] = results[1]
            for deposit_row in deposit_rows:
                raw_links[withdraw_row.hash] = deposit_row.hash

                graph.add_node(withdraw_row.hash)
                graph.add_node(deposit_row.hash)
                graph.add_edge(withdraw_row.hash, deposit_row.hash)

                # save transaction -> address map
                tx2addr[withdraw_row.hash] = withdraw_row.from_address
                tx2addr[deposit_row.hash] = deposit_row.from_address

    clusters: List[Set[str]] = [  # ignore singletons
        c for c in nx.weakly_connected_components(graph) if len(c) > 1]

    print(f'# links (graph): {len(clusters)}')
    print(f'# links (raw): {len(raw_links)}')

    return clusters, tx2addr


def exact_match_heuristic(
    deposit_df: pd.DataFrame,
    withdraw_df: pd.DataFrame,
) -> Tuple[bool, Optional[List[pd.Series]]]:
    """
    matches: List[pd.Series] = []
    # iterate over each deposit transaction. When a matching deposit is found, 
    # its hash is pushed to the list same_deposit_address_hashes.
    for deposit_row in deposit_df.itertuples():
        # check that addresses are the same and that the deposit 
        # was done earlier than the withdraw.
        if ((deposit_row.from_address == withdraw_df.recipient_address) and 
            (deposit_row.block_timestamp < withdraw_df.block_timestamp)):
            matches.append(deposit_row)
    """
    matches: pd.DataFrame = deposit_df[
        (deposit_df.from_address == withdraw_df.recipient_address) &
        (deposit_df.block_timestamp < withdraw_df.block_timestamp)
    ]
    matches: List[pd.Series] = [matches.iloc[i] for i in range(len(matches))]

    if len(matches) > 0:
        return (True, matches)

    return (False, None)


def exact_match_heuristic_by_pool(
    deposit_df: pd.DataFrame,
    withdraw_df: pd.DataFrame,
) -> Tuple[bool, Optional[List[pd.Series]]]:
    matches: pd.DataFrame = deposit_df[
        (deposit_df.from_address == withdraw_df.recipient_address) &
        (deposit_df.block_timestamp < withdraw_df.block_timestamp) & 
        (deposit_df.tornado_cash_address == withdraw_df.tornado_cash_address)
    ]
    matches: List[pd.Series] = [matches.iloc[i] for i in range(len(matches))]

    if len(matches) > 0:
        return (True, matches)

    return (False, None)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to tornado cash data')
    parser.add_argument('save_dir', type=str, help='folder to save matches')
    parser.add_argument('--by-pool', action='store_true', default=False,
                        help='prune by pool heuristic or not?')
    args: Any = parser.parse_args()

    main(args)
