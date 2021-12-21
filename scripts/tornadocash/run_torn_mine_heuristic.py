import os
import pandas as pd
from tqdm import tqdm
import networkx as nx
from typing import Any, Tuple, Dict, List, Set

from src.utils.utils import to_json, from_json, Entity, Heuristic

MINE_POOL_RATES: Dict[str, int] ={
    '0.1 ETH': 4, 
    '1 ETH': 20, 
    '10 ETH': 50, 
    '100 ETH': 400, 
    '100 DAI': 2, 
    '1000 DAI': 10, 
    '10000 DAI': 40, 
    '100000 DAI': 250, 
    '5000 cDAI': 2, 
    '50000 cDAI': 10,
    '500000 cDAI': 40, 
    '5000000 cDAI': 250,
    '0.1 WBTC': 15, 
    '1 WBTC': 120, 
    '10 WBTC': 1000,
}
MINE_POOL: List[str] = list(MINE_POOL_RATES.keys())


def main(args: Any):
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)

    cache_file: str = os.path.join(args.data_dir, 'heuristic_5_linked_txs.json')

    if os.path.isfile(cache_file):
        total_linked_txs: Dict[str, Dict[str, Any]] = from_json(cache_file)
    else:
        deposit_df, withdraw_df, miner_df = load_data(args.data_dir)

        deposit_df: pd.DataFrame = deposit_df[deposit_df['tcash_pool'].isin(MINE_POOL)]
        withdraw_df: pd.DataFrame = withdraw_df[withdraw_df['tcash_pool'].isin(MINE_POOL)]

        unique_deposits: Set[str] = set(deposit_df['from_address'])
        unique_withdraws: Set[str] = set(withdraw_df['recipient_address'])

        addr2deposits: Dict[str, Any] = address_to_txs_and_blocks(deposit_df, 'deposit')
        addr2withdraws: Dict[str, Any] = address_to_txs_and_blocks(withdraw_df, 'withdraw')

        total_linked_txs: Dict[str, Dict[str, Any]] = get_total_linked_txs(
            miner_df, unique_deposits, unique_withdraws, addr2deposits, addr2withdraws)

    w2d: Dict[Tuple[str], List[Tuple[str]]] = \
        apply_anonymity_mining_heuristic(total_linked_txs)

    clusters, tx2addr = build_clusters(w2d)
    address_sets: List[Set[str]] = get_address_sets(clusters, tx2addr)
    address_metadata: List[Dict[str, Any]] = get_metadata(address_sets)

    clusters_file: str = os.path.join(args.save_dir, 'torn_mine_clusters.json')
    tx2addr_file: str = os.path.join(args.save_dir, 'torn_mine_tx2addr.json')
    address_file: str = os.path.join(args.save_dir, 'torn_mine_address_set.json')
    metadata_file: str = os.path.join(args.save_dir, 'torn_mine_metadata.csv')
    to_json(clusters, clusters_file)
    to_json(tx2addr, tx2addr_file)
    to_json(address_sets, address_file)
    address_metadata.to_csv(metadata_file, index=False)


def build_clusters(links: Any) -> Tuple[List[Set[str]], Dict[str, str]]:
    graph: nx.DiGraph = nx.DiGraph()
    tx2addr: Dict[str, str] = {}

    for withdraw_tuple, deposit_tuples in links.items():
        withdraw_tx, withdraw_addr, _ = withdraw_tuple
        graph.add_node(withdraw_tx)
        tx2addr[withdraw_tx] = withdraw_addr

        for deposit_tuple in deposit_tuples:
            deposit_tx, deposit_addr = deposit_tuple
            graph.add_node(deposit_tx)
            tx2addr[deposit_tx] = deposit_addr

    clusters: List[Set[str]] = [  # ignore singletons
        c for c in nx.weakly_connected_components(graph) if len(c) > 1]

    return clusters, tx2addr


def get_address_sets(
    tx_clusters: List[Set[str]],
    tx2addr: Dict[str, str],
) -> List[Set[str]]:
    """
    Stores pairs of addresses that are related to each other. Don't 
    apply graphs on this because we are going to join this into the 
    other clusters.
    """
    address_sets: List[Set[str]] = []

    for cluster in tx_clusters:
        addr_set: Set[str] = set([tx2addr[tx] for tx in cluster])
        addr_set: List[str] = list(addr_set)

        if len(addr_set) > 1:  # make sure not singleton
            for addr1 in addr_set:
                for addr2 in addr_set:
                    if addr1 != addr2:
                        address_sets.append({addr1, addr2})

    return address_sets


def get_metadata(address_sets: List[Set[str]]) -> pd.DataFrame:
    """
    Stores metadata about addresses to add to db. 
    """
    unique_addresses: Set[str] = set().union(*address_sets)

    address: List[str] = []
    entity: List[int] = [] 
    conf: List[float] = []
    meta_data: List[str] = []
    heuristic: List[int] = []

    for member in unique_addresses:
        address.append(member)
        entity.append(Entity.EOA.value)
        conf.append(1)
        meta_data.append(json.dumps({}))
        heuristic.append(Heuristic.TORN_MINE.value)

    response: Dict[str, List[Any]] = dict(
        address = address,
        entity = entity,
        conf = conf,
        meta_data = meta_data,
        heuristic = heuristic,
    )
    response: pd.DataFrame = pd.DataFrame.from_dict(response)
    return response


def load_data(root) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tornado_addrs: Dict[str, str] = from_json(os.path.join(root, 'tornado_pools.json'))

    deposit_df: pd.DataFrame = pd.read_csv(
        os.path.join(root, 'lighter_complete_deposit_txs.csv'))
    deposit_df['tcash_pool'] = deposit_df['tornado_cash_address'].apply(
        lambda addr: tornado_addrs[addr])

    withdraw_df: pd.DataFrame = pd.read_csv(
        os.path.join(root, 'lighter_complete_withdraw_txs.csv'))
    withdraw_df['tcash_pool'] = withdraw_df['tornado_cash_address'].apply(
        lambda addr: tornado_addrs[addr])

    miner_df: pd.DataFrame = pd.read_csv(os.path.join(root, 'lighter_miner_txs.csv'))
    miner_df: pd.DataFrame = miner_df[miner_df['function_call'] == 'w']

    return deposit_df, withdraw_df, miner_df


def address_to_txs_and_blocks(txs_df: pd.DataFrame, tx_type: str) -> Dict[str, Any]:
    assert tx_type in ['deposit', 'withdraw'], 'Transaction type error'
    address_field: str = 'from_address' if tx_type == 'deposit' else 'recipient_address'
    addr_to_txs_and_blocks: Dict[str, Any] = {}

    for _, row in tqdm(txs_df.iterrows(), total=len(txs_df)):
        if row[address_field] not in addr_to_txs_and_blocks.keys():
            addr_to_txs_and_blocks[row[address_field]] = \
                {row.tcash_pool: [(row.hash, row.block_number)]}
        elif row.tcash_pool not in addr_to_txs_and_blocks[row[address_field]].keys():
            addr_to_txs_and_blocks[row[address_field]].update(
                {row.tcash_pool: [(row.hash, row.block_number)]})
        else:
            addr_to_txs_and_blocks[row[address_field]][row.tcash_pool].append(
                (row.hash, row.block_number))
    
    return addr_to_txs_and_blocks


'''
To classify the addresses by their inclusion in the unique_deposit_addresses and 
the unique_withdraw_addresses sets.
'''

def is_D_type(address: str, deposits: Set[str], withdraws: Set[str]):
    return (address in deposits) and (address not in withdraws)


def is_W_type(address: str, deposits: Set[str], withdraws: Set[str]):
    return (address not in deposits) and (address in withdraws)


def is_DW_type(address: str, deposits: Set[str], withdraws: Set[str]):
    return (address in deposits) and (address in withdraws)


def ap2blocks(anonymity_points: int, pool: str) -> float:
    rate = MINE_POOL_RATES[pool]
    return anonymity_points / float(rate)


def D_type_anonymity_heuristic(
    miner_tx: pd.Series, 
    addr2deposits: Dict[str, Any],
    addr2withdraws: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    d_addr: str = miner_tx.recipient_address
    d_addr2w: Dict[str, Dict[str, Any]] = {d_addr: {}}

    breakpoint()

    for d_pool in addr2deposits[d_addr]:
        for (d_hash, d_blocks) in addr2deposits[d_addr][d_pool]:
            delta_blocks: float = ap2blocks(miner_tx.anonimity_points, d_pool)

            for w_addr in addr2withdraws.keys():
                if d_pool in addr2withdraws[w_addr].keys():
                    for (w_hash, w_blocks) in addr2withdraws[w_addr][d_pool]:
                        if d_blocks + delta_blocks == w_blocks:
                            if d_hash not in d_addr2w[d_addr].keys():
                                d_addr2w[d_addr][d_hash] = [(w_hash, w_addr, delta_blocks)]
                            else:
                                d_addr2w[d_addr][d_hash].append((w_hash, w_addr, delta_blocks))

    return d_addr2w


def W_type_anonymity_heuristic(
    miner_tx: pd.Series, 
    addr2deposits: Dict[str, Any],
    addr2withdraws: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    w_addr: str = miner_tx.recipient_address
    w_addr2d: Dict[str, Dict[str, Any]] = {w_addr: {}}
    
    for w_pool in addr2withdraws[w_addr]:
        for (w_hash, w_blocks) in addr2withdraws[w_addr][w_pool]:
            delta_blocks: float = ap2blocks(miner_tx.anonimity_points, w_pool)

            for d_addr in addr2deposits.keys():
                if w_pool in addr2deposits[d_addr].keys():
                    for (d_hash, d_blocks) in addr2deposits[d_addr][w_pool]:
                        if d_blocks + delta_blocks == w_blocks:
                            if w_hash not in w_addr2d[w_addr].keys():
                                w_addr2d[w_addr][w_hash] = [(d_hash, d_addr, delta_blocks)]
                            else:
                                w_addr2d[w_addr][w_hash].append((d_hash, d_addr, delta_blocks))

    return w_addr2d


def anonymity_mining_heuristic(
    miner_tx: pd.Series,
    unique_deposits: Set[str],
    unique_withdraws: Set[str],
    addr2deposits: Dict[str, Any],
    addr2withdraws: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    linked_txs: Dict[str, Dict[str, Any]] = {}

    if is_D_type(miner_tx.recipient_address, unique_deposits, unique_withdraws):
        d_dict: Dict[str, Any] = D_type_anonymity_heuristic(
            miner_tx, addr2deposits, addr2withdraws)
        if len(d_dict[miner_tx.recipient_address]) != 0:
            linked_txs['D'] = d_dict
        return linked_txs
    elif is_W_type(miner_tx.recipient_address, unique_deposits, unique_withdraws):
        w_dict: Dict[str, Any] = W_type_anonymity_heuristic(
            miner_tx, addr2deposits, addr2withdraws)
        if len(w_dict[miner_tx.recipient_address]) != 0:
            linked_txs['W'] = w_dict
        return linked_txs
    elif is_DW_type(miner_tx.recipient_address, unique_deposits, unique_withdraws):
        d_dict: Dict[str, Any] = D_type_anonymity_heuristic(
            miner_tx, addr2deposits, addr2withdraws)
        if len(d_dict[miner_tx.recipient_address]) != 0:
            linked_txs['D'] = d_dict
        w_dict: Dict[str, Any] = W_type_anonymity_heuristic(
            miner_tx, addr2deposits, addr2withdraws)
        if len(w_dict[miner_tx.recipient_address]) != 0:
            linked_txs['W'] = w_dict
        return linked_txs

    return linked_txs


def get_total_linked_txs(
    miner_txs: pd.Series,
    unique_deposits: Set[str],
    unique_withdraws: Set[str],
    addr2deposits: Dict[str, Any], 
    addr2withdraws: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    total_linked_txs: Dict[str, Dict[str, Any]] = {'D': {}, 'W': {}}

    for miner_tx in tqdm(miner_txs.itertuples(), total=len(miner_txs)):
        linked_txs: Dict[str, Dict[str, Any]] = anonymity_mining_heuristic(
            miner_tx, unique_deposits, unique_withdraws, addr2deposits, addr2withdraws)
        if len(linked_txs) != 0:
            if 'D' in linked_txs.keys():
                if len(linked_txs['D']) != 0:
                    total_linked_txs['D'].update(linked_txs['D'])
            if 'W' in linked_txs.keys():
                if len(linked_txs['W']) != 0:
                    total_linked_txs['W'].update(linked_txs['W'])

    return total_linked_txs


def apply_anonymity_mining_heuristic(
    total_linked_txs: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[str], List[Tuple[str]]]:
    """
    The final version of the results is obtained applying this function
    to the output of the 'apply_anonimity_mining_heuristic' function.

    w2d -> withdraws and blocks to deposits
    """
    w2d: Dict[Tuple[str], List[Tuple[str]]] = {}

    for addr in total_linked_txs['W'].keys():
        for hsh in total_linked_txs['W'][addr]:
            delta_blocks: float = total_linked_txs['W'][addr][hsh][0][2]
            w2d[(hsh, addr, delta_blocks)] = [
                (t[0],t[1]) for t in total_linked_txs['W'][addr][hsh]]

    for addr in total_linked_txs['D'].keys():
        for hsh in total_linked_txs['D'][addr]:
            for tx_tuple in total_linked_txs['D'][addr][hsh]:
                if tx_tuple[0] not in w2d.keys():
                    w2d[tuple(tx_tuple)] = [(hsh, addr)]
                else:
                    if (hsh, addr) not in w2d[tx_tuple]:
                        w2d[tuple(tx_tuple)].append((hsh, addr))
    return w2d


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to tornado cash data')
    parser.add_argument('save_dir', type=str, help='folder to save matches')
    args: Any = parser.parse_args()

    main(args)
