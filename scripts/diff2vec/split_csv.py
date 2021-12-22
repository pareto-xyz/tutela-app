"""
Split giant CSV into many small csvs
"""
import os, sys, csv
from typing import List

csv.field_size_limit(sys.maxsize)


def split(file_loc: str, out_dir: str, file_size=100000):
    with open(file_loc) as fp:
        count: int = 0
        curr_split: int = 0
        reader = csv.reader(fp)
        for header in reader: break
        writer = None

        for row in reader:
            if count % file_size == 0:
                print(f'parsed {count} rows.')
                split_filename: str = os.path.join(out_dir, f'edges-{curr_split}.csv')
                writer = csv.writer(open(split_filename, 'w'))
                writer.writerow(header)
                curr_split += 1

            writer.writerow(row)
            count += 1


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('edges_file', type=str, help='path to csv file containing edges')
    parser.add_argument('split_dir', type=str, help='path to dump split files')
    parser.add_argument('--file-size', type=int, default=100000)
    args = parser.parse_args()

    split(args.edges_file, args.split_dir, file_size=args.file_size)
