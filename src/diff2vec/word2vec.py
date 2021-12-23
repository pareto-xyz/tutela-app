"""
We want to use as much of GenSim's word2vec as possible for it's preprocessing
takes up too much memory for us. Let's see if we can reduce it.
"""
from gensim.models import Word2Vec