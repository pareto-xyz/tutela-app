from typing import Any
from src.cluster.lastchunk import restore_last_chunk


def main(args: Any):
    restore_last_chunk(args.transaction_csv, args.out_csv, min_block=args.min_block, t_max=args.t_max)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('transaction_csv', type=str, help='path to transaction csv')
    parser.add_argument('out_csv', type=str, help='path to output csv')
    parser.add_argument('min_block', type=int, help='smallest block number to consider')
    parser.add_argument('--t-max', type=int, default=3200, help='t_max (default: 3200)')
    args: Any = parser.parse_args()
    main(args)
