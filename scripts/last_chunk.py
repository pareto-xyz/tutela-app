from typing import Any
from src.cluster.lastchunk import restore_last_chunk


def main(args: Any):
    restore_last_chunk(args.transaction_csv, args.out_csv, chunk_size=args.chunk_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('transaction_csv', type=str, help='path to transaction csv')
    parser.add_argument('out_csv', type=str, help='path to output csv')
    parser.add_argument('--chunk-size', type=int, default=32000000, help='chunk size (default: 32M)')
    args: Any = parser.parse_args()
    main(args)
