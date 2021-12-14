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

    def _diffuse(self, node: int) -> List[int]:
        """
        Generate diffusion tree from source node.
        """
        infected: List[int] = [node]

        subgraph = nx.DiGraph()  # subgraphs we assume are small enough for nx
        subgraph.add_node(node)  # start with such this node
        counter: int = 1

        while counter < self.subgraph_size:
            w: int = self.rs.choice(infected)
            neighbors: List[int] = self.graph.neighbors(w)
            u: int = self.rs.choice(neighbors)
            if u not in infected:
                counter += 1
                infected.append(u)
                # double graph
                subgraph.add_edge(u, w)

                if counter == self.subgraph_size:
                    break

        euler: List[int] = [u for u,_ in nx.eulerian_circuit(subgraph, node)]
        return euler

    def diffuse(self) -> Dict[int, List[int]]:
        circuit: Dict[int, List[int]] = {}
        for node in self.graph.nodes():
            seq: List[int] = self._diffuse(node)
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

    def get_sequences(self) -> List[List[int]]:
        print('Computing connected components')
        subgraphs: List[UndirectedGraph] = self.extract_components(self.graph)
        paths: Dict[int, List[int]] = dict() 

        pbar = tqdm(total=len(subgraphs))
        for subgraph in subgraphs:
            card: int = len(subgraph)  # cardinality
            if card < self.vertex_card:
                self.vertex_card: int = card

            euler: EulerianDiffusion = \
                EulerianDiffusion(subgraph, self.vertex_card)
            circuits: Dict[int, List[int]] = euler.diffuse()

            paths.update(circuits)
            pbar.update()

        pbar.close()
        paths = [v for _, v in paths.items()]
        return paths

    def get_count(self):
        return len(self.graph) + 1
