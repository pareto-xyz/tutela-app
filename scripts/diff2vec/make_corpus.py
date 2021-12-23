"""
Convert the files to a gensim corpus style.

gensim LineSentence format: 

`one line = one sentence. Words must be already preprocessed and 
separated by whitespace.`
"""
import os
import jsonlines
from glob import glob
from typing import Any, List


def main(args: Any):
    corpus_file: str = os.path.join(args.data_dir, 'corpus-30.jsonl')
    with jsonlines.open(corpus_file, 'w') as out_fp:
        sequence_files: List[str] = glob(
            os.path.join(args.data_dir, 'sequences-30-*.jsonl'))
        sizes: List[int] = [0] * len(sequence_files)
        count: int = 0
        for i, sequence_file in enumerate(sequence_files):
            sequence_file: str = sequence_file
            with jsonlines.open(sequence_file) as in_fp: 
                for row in in_fp:
                    row: List[int] = row
                    out_fp.write(row)
                    sizes[i] += 1
                    count += 1

                    if count % 1000000 == 0:
                        print(f'Written {count} files.')

        print(sizes)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to data_dir')
    args: Any = parser.parse_args()

    main(args)
