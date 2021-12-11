"""
Helper functions to run Diff2Vec on a NetworkX graph.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple
from gensim.models import Word2Vec

from src.diff2vec.euler import SubGraphSequences


class Diff2Vec:
    """
    Adapted from https://github.com/benedekrozemberczki/karateclub

    An implementation of `"Diff2Vec" <http://homepages.inf.ed.ac.uk/s1668259/papers/sequence.pdf>`_
    from the CompleNet '18 paper "Diff2Vec: Fast Sequence Based Embedding with Diffusion Graphs".
    The procedure creates diffusion trees from every source node in the graph. These graphs are linearized
    by a directed Eulerian walk, the walks are used for running the skip-gram algorithm the learn node
    level neighbourhood based embeddings.
    """

    def __init__(
        self,
        dimensions: int = 128,        # Dimensionality of the word vectors.
        window_size: int = 10,        # Maximum distance between the current and predicted word within a sentence.
        cover_size: int = 80,         # Number of nodes in diffusion.
        epochs: int = 1,              # Number of iterations (epochs) over the corpus.
        learning_rate: float = 0.05,  # The initial learning rate.
        workers: int = 4,             # Number of workers
        min_count: int = 1,           # Ignores all words with total frequency lower than this.
        seed: int = 42,               # Seed for the random number generator.
    ):
        self.window_size = window_size
        self.cover_size = cover_size
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def fit(self, graph: nx.Graph):
        """
        Fitting a Diff2Vec model.
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        sequencer: SubGraphSequences = SubGraphSequences(graph, self.cover_size)
        sequences: List[str] = sequencer.get_sequences()

        model: Word2Vec = Word2Vec(
            sequences,
            vector_size = self.dimensions,
            window = self.window_size,
            min_count = self.min_count,
            hs = 1,
            workers = self.workers,
            epochs = self.epochs,
            alpha = self.learning_rate,
            seed = self.seed,
        )

        num_nodes: int = graph.number_of_nodes()
        self._embedding = [model.wv[str(n)] for n in range(num_nodes)]

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.
        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)
