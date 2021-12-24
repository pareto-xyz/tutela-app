import os
import numpy as np
from typing import Any, List
from scipy.stats import gaussian_kde


def main(args: Any):
    vectors: np.array = np.load(args.vectors_npy)
    kde = gaussian_kde(vectors.T)
    density: np.array = kde(vectors.T)

    filename: str = os.path.basename(args.vectors_npy)
    name_pieces: List[str] = filename.split('.')
    name_pieces[0] = f'{name_pieces[0]}-density'
    filename = '.'.join(name_pieces)

    dirname: str = os.path.dirname(args.vectors_npy)
    np.save(os.path.join(dirname, filename), density)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('vectors_npy', type=str, 
                        help='path to trained word2vec vectors.')
    args: Any = parser.parse_args()

    main(args)
