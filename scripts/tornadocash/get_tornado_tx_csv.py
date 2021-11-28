"""
Create a CSV that can be uploaded directly into TornadoPool table. 
Looks over all interactions and records which interactions are depositing
into which tornado pools?
"""
from typing import Any


def main(args: Any):
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('deposit_csv', type=str, help='path to tornado cash deposit data')
    parser.add_argument('out_csv', type=str, help='where to save output file?')
    args: Any = parser.parse_args()

    main(args)
