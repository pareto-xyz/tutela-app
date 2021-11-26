"""
Lambda Class's Same-Number-Deposits Heuristic.
"""
import os
import pandas as pd
from tqdm import tqdm
import networkx as nx
from typing import Any, Tuple, List, Set, Dict
from src.utils.utils import to_json


def main(args: Any):
    withdraw_df, deposit_df, tornado_df = load_data(args.data_dir)
    clusters, tx2addr = get_same_num_deposits_clusters(
        deposit_df, withdraw_df, tornado_df)
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    clusters_file: str = os.path.join(args.save_dir, f'num_deposits_clusters.json')
    tx2addr_file: str = os.path.join(args.save_dir, f'num_deposits_tx2addr.json')
    to_json(clusters, clusters_file)
    to_json(tx2addr, tx2addr_file)


def load_data(root) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    # Load Tornado data
    tornado_df: pd.DataFrame = pd.read_csv(args.tornado_csv)

    return withdraw_df, deposit_df, tornado_df


def get_same_num_deposits_clusters(
    deposit_df: pd.DataFrame, 
    withdraw_df: pd.DataFrame, 
    tornado_df: pd.DataFrame,
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
    tornado_addresses: Dict[str, int] = \
        dict(zip(tornado_df.address, tornado_df.tags))

    addr2deposit: Dict[str, str] = \
        get_address_deposits(deposit_df, tornado_addresses)

    # initialize an empty dictionary
    tx2addr: Dict[str, str] = {}
    graph: nx.DiGraph = nx.DiGraph()

    for withdraw_row in withdraw_df.itertuples():
        results = same_number_of_deposits_heuristic(
            withdraw_row, addr2deposit, withdraw_df, tornado_addresses)

        if results[0]:
            deposit_rows: List[pd.Series] = results[1]
            for deposit_row in deposit_rows:
                graph.add_node(withdraw_row.hash)
                graph.add_node(deposit_row.hash)
                graph.add_edge(withdraw_row.hash, deposit_row.hash)

                # save transaction -> address map
                tx2addr[withdraw_row.hash] = withdraw_row.from_address
                tx2addr[deposit_row.hash] = deposit_row.from_address

    clusters: List[Set[str]] = [  # ignore singletons
        c for c in nx.weakly_connected_components(graph) if len(c) > 1]

    return clusters, tx2addr


def same_number_of_deposits_heuristic(
    withdraw_tx: pd.Series, 
    addr2deposit: Dict[str, str], 
    withdraw_df: pd.DataFrame, 
    tornado_addresses: Dict[str, int],
) -> List[str]:
    # Calculate the number of withdrawals of the address 
    # from the withdraw_tx given as input.
    withdraw_counts: pd.DataFrame = get_number_of_withdraws(
        withdraw_tx, withdraw_df, tornado_addresses)

    # Based on n_withdraws, the set of the addresses that have 
    # the same number of deposits is calculated.
    matches: List[str] = get_same_or_more_number_of_deposits(
        withdraw_counts, addr2deposit, tornado_addresses)

    if len(matches) > 0:
        return (True, matches)

    return (False, None)


def get_same_or_more_number_of_deposits(
    withdraw_counts: pd.DataFrame, 
    addr2deposit: Dict[str, str], 
) -> List[str]:
    result: Dict[str, Any] = dict(
        filter(lambda elem: compare_transactions(withdraw_counts, elem[1]), 
                addr2deposit.items()))
    return list(result.keys())


def compare_transactions(
    withdraw_dict: pd.DataFrame, 
    deposit_dict: pd.DataFrame,
) -> bool:
    """
    Given two dictionaries, withdraw_dict and deposit_dict representing 
    the total deposits and withdraws made by an address to each TCash pool, 
    respectively, compares if the set of keys of both are equal and when 
    they are, checks if all values in the deposit dictionary are equal or 
    greater than each of the corresponding values of the withdraw 
    dicionary. If this is the case, returns True, if not, False.
    """
    if set(withdraw_dict.keys()) != set(deposit_dict.keys()):
        return False
    for currency in withdraw_dict.keys():
        if not (deposit_dict[currency] >= withdraw_dict[currency]):
            return False
    return True


def get_number_of_withdraws(
    withdraw_tx: pd.Series, 
    withdraw_df: pd.DataFrame, 
    tornado_addresses: Dict[str, str],
) -> Dict[str, int]:
    """
    Given a particular withdraw transaction and the withdraw transactions 
    DataFrame, gets the total withdraws the address made in each pool. It 
    is returned as a dictionary with the pools as the keys and the number 
    of withdraws as the values.
    """
    withdraw_count: Dict[str, int] = {
        tornado_addresses[withdraw_tx.tornado_cash_address]: 1}

    for withdraw_row in withdraw_df.itertuples():
        if ((withdraw_row.recipient_address == withdraw_tx.recipient_address) and 
            (withdraw_row.block_timestamp <= withdraw_tx.block_timestamp) and 
            (withdraw_row.hash != withdraw_tx.hash)):

            if tornado_addresses[
                withdraw_row.tornado_cash_address] in withdraw_count.keys():
                withdraw_count[
                    tornado_addresses[withdraw_row.tornado_cash_address]] += 1
            else:
                withdraw_count[
                    tornado_addresses[withdraw_row.tornado_cash_address]] = 1
    return withdraw_count


def get_address_deposits(
    deposit_df: pd.DataFrame,
    tornado_addresses: Dict[str, int],
) -> Dict[str, str]:
    """
    Given the deposit transactions DataFrame, returns a 
    dictionary with every address of the deposit

    Example:
    {
        '0x16e54b35d789832440ab47ae765e6a8098280676': 
            {
                '0.1 ETH': 1068,
                '100 USDT': 1,
            },
        '0x35dd029618f4e1835008da21fd98850c776453f0': {
            '0.1 ETH': 628,
        },
        '0xe906442c11b85acbc58eccb253b9a55a20b80a56': {
            '0.1 ETH': 562,
        },
        '0xaf301de836c81deb8dff9dc22745e23c476155b2': {
            '1 ETH': 503,
            '0.1 ETH': 96,
            '10 ETH': 8,
        },
    }
    """
    counts_df: pd.DataFrame = pd.DataFrame(
        deposit_df[['from_address', 'tornado_cash_address']].value_counts()
    ).rename(columns={0: "count"})
    
    addr2deposit: Dict[str, str] = {}
    for row in counts_df.itertuples():
        if row.Index[0] in addr2deposit.keys():
            addr2deposit[row.Index[0]][
                tornado_addresses[row.Index[1]]] = row.count
        else:
            addr2deposit[row.Index[0]] = {
                tornado_addresses[row.Index[1]]: row.count}

    return addr2deposit


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to tornado cash data')
    parser.add_argument('tornado_csv', type=str, help='path to tornado cash pool addresses')
    parser.add_argument('save_dir', type=str, help='folder to save matches')
    args: Any = parser.parse_args()

    main(args)
