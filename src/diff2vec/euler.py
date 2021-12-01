"""
Eulerian Diffusion.

This file depends on NetworkX, which I am not sure is scalable
enough to do what we want it to do.
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Any


class EulerianDiffusion:
    """
    Algorithm 1 in https://arxiv.org/pdf/2001.07463.pdf.

    This computes an Eulerian sequence from a Graph object. 
    The parameter `num_nodes` represents the size of subgraph 
    to be created; also denoted by `l` in the paper.
    """

    def __init__(self, graph: nx.Graph, num_nodes: int):
        self.graph: nx.Graph = graph
        self.num_nodes: int = num_nodes
        self.rs = np.random.RandomState(42)

    def _diffuse(self, node: str) -> List[str]:
        """
        Generate diffusion tree from source node.
        """
        infected: List[str] = [node]

        subgraph = nx.DiGraph()
        subgraph.add_node(node)  # start with such this node
        counter: int = 1

        while counter < self.num_nodes:
            w: str = self.rs.choice(infected)
            neighbors: List[str] = [node for node in self.graph.neighbors(w)]
            u: str = self.rs.choice(neighbors)
            if u not in infected:
                counter += 1
                infected.append(u)
                # double graph
                subgraph.add_edges_from([(w, u), (u, w)])

                if counter == self.num_nodes:
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
    def __init__(self, graph: nx.Graph, vertex_card: int):
        self.graph: nx.Graph = graph
        self.vertex_card = vertex_card  # number of nodes per sample

    def extract_components(self, graph):
        components: List[Any] = [
            graph.subgraph(c) for c in nx.connected_components(graph)]
        # sort from biggest component to smallest
        components: List[Any] = sorted(components, key=len, reverse=True)
        return components

    def get_sequences(self):
        subgraphs: Any = self.extract_components(self.graph)
        paths: Dict[str, List[str]] = dict() 

        for subgraph in subgraphs:
            card: int = len(subgraph.nodes())  # cardinality
            if card < self.vertex_card:
                self.vertex_card: int = card

            euler: EulerianDiffusion = \
                EulerianDiffusion(subgraph, self.vertex_card)
            circuits: Dict[str, List[str]] = euler.diffuse()

            paths.update(circuits)

        paths = [v for k, v in paths.items()]
        return paths
