"""
We can't store the Word2Vec in RAM on the server so we should 
instead store a map from address index -> clusters of indices.
"""
import os
import faiss
import numpy as np
from typing import Any, List
from tqdm import tqdm


def main(args: Any):
    print('loading vectors...', end=' ')
    vectors: np.array = np.load(args.vectors_npy)
    print('done')
    size = vectors.shape[0]

    # https://github.com/facebookresearch/faiss/issues/112
    nlist = 4 * np.sqrt(size)

    quantizer: faiss.IndexFlatL2 = faiss.IndexFlatL2(128)
    index: faiss.IndexIVFFlat = \
        faiss.IndexIVFFlat(quantizer, 128, nlist, faiss.METRIC_L2)

    assert not index.is_trained
    print('training FAISS index...', end=' ')
    index.train(vectors)
    print('done')
    assert index.is_trained

    print('adding vectors to index...', end=' ')
    index.add(vectors)
    print('done')

    del vectors  # free up space

    print('computing neighbors')
    distances: List[np.array] = []
    neighbors: List[np.array] = []

    batch_size: int = 100
    num_batches: int = (size // batch_size) + int(size % batch_size)

    for i in tqdm(range(num_batches)):
        query: np.array = vectors[batch_size*i:batch_size*(i+1)]
        D, I = index.search(query)
        distances.append(D)
        neighbors.append(I)

    distances = np.concatenate(distances, axis=0)
    neighbors = np.concatenate(neighbors, axis=0)

    np.save(os.path.join(args.save_dir, f'distances-k{args.k}.npy'), distances)
    np.save(os.path.join(args.save_dir, f'neighbors-k{args.k}.npy'), neighbors)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('vectors_npy', type=str, help='path to trained word2vec vectors.')
    parser.add_argument('save_dir', type=str, help='where to save outpouts.')
    parser.add_argument('--k', type=int, default=10, help='number of neighbors to find.')
    args: Any = parser.parse_args()

    main(args)