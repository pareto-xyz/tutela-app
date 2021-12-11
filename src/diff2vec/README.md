# Diff2Vec algorithm: 

Original Paper: https://arxiv.org/pdf/2001.07463.pdf [1]
Istvan's Paper: https://arxiv.org/pdf/2005.14051.pdf [2]

The set of addresses used in interactions characterize a user. Users with multiple accounts might interact with the same addresses or services from most of them. Furthermore, as users move funds between their personal addresses, they may unintentionally reveal their address clusters.

Our deanonymization experiments are conducted on a transaction graph with nodes as Ethereum addresses and edges as transactions.

Preprocessing:
- Transactions are treated as undirected edges.
- Loops and multi-edges are removed. 
- Remove nodes with degree one.

Our edits:
- Store a weight representing multi edges. I think this is important for a good representation. We can also build this weight into the frequency algorithm pretty easily (just count number of interactions as frequency).
- Also, this has never been done at scale. [2] only does 16k nodes from Ethereum.
