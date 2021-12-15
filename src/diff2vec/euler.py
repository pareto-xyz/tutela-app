"""
Eulerian Diffusion.
"""
import random
import json, jsonlines
from tqdm import tqdm
import networkx as nx
from typing import List, Dict, Set, Tuple
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

    def _diffuse(self, node: int) -> List[int]:
        """
        Generate diffusion tree from source node.
        """
        infected: Set[int] = {node}

        subgraph = nx.DiGraph()  # subgraphs we assume are small enough for nx
        subgraph.add_node(node)  # start with such this node
        counter: int = 1

        while counter < self.subgraph_size:
            w: int = random.sample(infected, 1)[0]
            neighbors: Set[int] = self.graph.neighbors(w)
            neighbors: Set[int] = neighbors - infected

            if len(neighbors) == 0:  # nothing to do!
                break

            # set subtract so we always sample a new node 
            # rather than rejection sample
            u: int = random.sample(neighbors, 1)[0]
            
            counter += 1
            infected.add(u)
            # double graph
            subgraph.add_edges_from([(u, w), (w, u)])

            if counter == self.subgraph_size:
                break

        euler: List[int] = [int(u) for u, _ in nx.eulerian_circuit(subgraph, node)]
        return euler

    def diffuse(self, writer: jsonlines.Writer, verbose: bool = False) -> Dict[int, List[int]]:
        if verbose: pbar = tqdm(total=len(self.graph))
        for node in self.graph.nodes():
            seq: List[int] = self._diffuse(node)
            writer.write(json.dumps((node, seq)))
            if verbose: pbar.update()
        if verbose: pbar.close()


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

    def get_sequences(self, out_file: str) -> Tuple[List[int], List[List[int]]]:
        print('Computing connected components')
        subgraphs: List[UndirectedGraph] = self.extract_components(self.graph)

        pbar = tqdm(total=len(subgraphs))
        with jsonlines.open(out_file, mode='w') as writer:
            for subgraph in subgraphs:
                card: int = len(subgraph)  # cardinality

                if card == 1:  # skip components of size 1
                    continue

                if card < self.vertex_card:
                    self.vertex_card: int = card

                euler: EulerianDiffusion = \
                    EulerianDiffusion(subgraph, self.vertex_card)
                euler.diffuse(writer, verbose=len(subgraph) > 10000)

            pbar.update()
        pbar.close()

    def get_count(self):
        return len(self.graph) + 1
