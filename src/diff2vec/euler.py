"""
Eulerian Diffusion.
"""
import numpy as np
from tqdm import tqdm
import networkx as nx
from typing import List, Dict, Set
from src.diff2vec.graph import UndirectedGraph


class EulerianDiffusion:
    """
    Algorithm 1 in https://arxiv.org/pdf/2001.07463.pdf.

    This computes an Eulerian sequence from a Graph object. 
    The parameter `subgraph_size` represents the size of subgraph 
    to be created; also denoted by `l` in the paper.

    Assumes all nodes in the graph are integers.
    """

    def __init__(self, graph: UndirectedGraph, subgraph_size: int):
        self.graph: UndirectedGraph = graph
        self.subgraph_size: int = subgraph_size
        self.rs = np.random.RandomState(42)

    def _diffuse(self, node: str) -> List[str]:
        """
        Generate diffusion tree from source node.
        """
        infected: List[int] = [node]

        subgraph = nx.DiGraph()  # subgraphs we assume are small enough for nx
        subgraph.add_node(node)  # start with such this node
        counter: int = 1

        while counter < self.subgraph_size:
            w: str = self.rs.choice(infected)
            neighbors: List[int] = self.graph.neighbors(w)
            u: str = self.rs.choice(neighbors)
            if u not in infected:
                counter += 1
                infected.append(u)
                # double graph
                subgraph.add_edges_from([(u, w), (w, u)])

                if counter == self.subgraph_size:
                    break

        euler: List[str] = [u for u,_ in nx.eulerian_circuit(subgraph, node)]
        return euler

    def diffuse(self) -> Dict[str, List[str]]:
        circuit: Dict[str, List[str]] = {}
        for node in self.graph.nodes():
            seq: List[str] = self._diffuse(node)
            circuit[node] = seq

        return circuit


class SubGraphSequences:
    """
    Algorithm 2 in https://arxiv.org/pdf/2001.07463.pdf.

    Separate the original graph and run diffusion on each node 
    in subgraph.
    """
    def __init__(self, graph: UndirectedGraph, vertex_card: int):
        self.graph: UndirectedGraph = graph
        self.vertex_card: int = vertex_card  # number of nodes per sample

    def extract_components(self, graph) -> List[UndirectedGraph]:
        # find subgraphs of network as separate graphs
        components: List[Set[int]] = graph.connected_components()
        components: List[UndirectedGraph] = \
            [graph.subgraph(c) for c in components]
        # sort from biggest component to smallest
        components: List[UndirectedGraph] = sorted(components, key=len, reverse=True)
        return components

    def get_sequences(self):
        print('Computing connected components')
        subgraphs: List[UndirectedGraph] = self.extract_components(self.graph)
        paths: Dict[str, List[str]] = dict() 

        pbar = tqdm(total=len(subgraphs))
        for subgraph in subgraphs:
            card: int = len(subgraph)  # cardinality

            if card == 1:  # skip components of size 1
                continue

            if card < self.vertex_card:
                self.vertex_card: int = card

            euler: EulerianDiffusion = \
                EulerianDiffusion(subgraph, self.vertex_card)
            circuits: Dict[str, List[str]] = euler.diffuse()

            paths.update(circuits)
            pbar.update()

        pbar.close()
        paths = [v for _, v in paths.items()]
        return paths

    def get_count(self):
        return len(self.graph) + 1
