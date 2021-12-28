import os
import numpy as np
from typing import Any, List
from sklearn.decomposition import PCA


def main(args: Any):
    vectors: np.array = np.load(args.vectors_npy)
    pca: PCA = PCA(n_components=3)
    vectors: np.array = pca.fit_transform(vectors)

    filename: str = os.path.basename(args.vectors_npy)
    name_pieces: List[str] = filename.split('.')
    name_pieces[0] = f'{name_pieces[0]}-pca'
    filename = '.'.join(name_pieces)

    dirname: str = os.path.dirname(args.vectors_npy)
    np.save(os.path.join(dirname, filename), vectors)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('vectors_npy', type=str, help='path to trained word2vec vectors.')
    args: Any = parser.parse_args()

    main(args)
