from typing import Any, Dict, List, Set
from src.utils.loader import DataframeLoader
from src.cluster.deposit import DepositCluster


def build_response(
    user_clusters: List[Set[str]],      # user -> deposit 
    exchange_clusters: List[Set[str]],  # deposit -> exchange
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Construct a mapping from an address to an index. That index 
    picks a set with the clusters list. Also store any extra metadata.
    """
    response = dict()

    for name, clusters in zip(
        ['user_cluster', 'exchange_cluster'],
        [user_clusters, exchange_clusters],
    ):
        for i, cluster in enumerate(clusters):
            cluster: List[Set[int]] = cluster

            for address in cluster:
                assert address in metadata, "error: unknown address"

                if address in response:
                    assert name not in response[address], \
                        "error: each element only allowed to be in one cluster."
                    response[address][name] = i
                else:
                    response[address] = {name: i, **metadata[address]}

    return response


def main(args: Any):
    if args.dataset == 'mini_bionic':
        loader: DataframeLoader = DataframeLoader(
            args.blocks_csv,
            args.known_addresses_csv,
            args.transactions_csv,
            args.save_dir,
        )
    elif args.dataset == 'bigquery':
        loader: DataframeLoader = DataframeLoader(
            args.blocks_csv,
            args.known_addresses_csv,
            args.transactions_csv,
            args.save_dir,
        )
    else:
        raise Exception(f'Dataset {args.dataset} not supported.')

    algo = DepositCluster(
        loader,
        a_max = args.a_max,
        t_max = args.t_max,
        save_dir = args.save_dir,
    )

    # this saves user/deposit/exchange columns but does not 
    # compute weakly connected components. See run_nx.py.
    algo.make_clusters()
    print('done.')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('blocks_csv', type=str, help='path to block data')
    parser.add_argument('transactions_csv', type=str, help='path to transaction data')
    parser.add_argument('known_addresses_csv', type=str, help='path to known address data')
    parser.add_argument('save_dir', type=str, help='path to save output')
    parser.add_argument('--dataset', type=str, default='bigquery',
                        choices=['mini_bionic', 'bigquery'], 
                        help='dataset name (default: mini_bionic)')
    parser.add_argument('--a-max', type=float, default=0.01, 
                        help='maximum amount difference (default: 0.01)')
    parser.add_argument('--t-max', type=float, default=3200,
                        help='maximum time difference (default: 3200)')
    args: Any = parser.parse_args()

    main(args)
