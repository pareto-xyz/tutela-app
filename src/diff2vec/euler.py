"""
Eulerian Diffusion.
"""
import os, random
import numpy as np
import json, jsonlines
from tqdm import tqdm
import networkx as nx
from typing import List, Dict, Set, Union
from src.diff2vec.graph import UndirectedGraph, UndirectedGraphCSV
from src.utils.utils import from_json


class EulerianDiffusion:
    """
    Algorithm 1 in https://arxiv.org/pdf/2001.07463.pdf.

    This computes an Eulerian sequence from a Graph object. 
    The parameter `cover_size` represents the size of subgraph 
    to be created; also denoted by `l` in the paper.

    Assumes all nodes in the graph are integers.
    """

    def __init__(
        self,
        graph: Union[UndirectedGraph, UndirectedGraphCSV],
        component: Set[int],
        cover_size: int):
        self.graph: Union[UndirectedGraph, UndirectedGraphCSV] = graph
        self.component: Set[int] = component
        self.cover_size: int = cover_size

    def _diffuse(self, node: int) -> List[int]:
        """
        Generate diffusion tree from source node.
        """
        infected: Set[int] = {node}

        subgraph: nx.DiGraph = nx.DiGraph()  # subgraphs we assume are small enough for nx
        subgraph.add_node(node)  # start with such this node
        counter: int = 1

        while counter < self.cover_size:
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

            if counter == self.cover_size:
                break

        euler: List[int] = [int(u) for u, _ in nx.eulerian_circuit(subgraph, node)]
        return euler

    def diffuse(
        self,
        writer: jsonlines.Writer,
        verbose: bool = False) -> Dict[int, List[int]]:
        if verbose: pbar = tqdm(total=len(self.component))
        for node in self.component:
            seq: List[int] = self._diffuse(node)
            writer.write(json.dumps(seq))
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

    def extract_components(self, graph: UndirectedGraph) -> List[UndirectedGraph]:
        # find subgraphs of network as separate graphs
        components: List[Set[int]] = graph.connected_components()
        # sort from biggest component to smallest
        components: List[Set[int]] = sorted(components, key=len, reverse=True)
        return components

    def get_sequences(self, out_file: str):
        print('Computing connected components')
        components: List[Set[int]] = self.extract_components(self.graph)

        pbar = tqdm(total=len(components))
        with jsonlines.open(out_file, mode='w') as writer:
            for component in components:
                card: int = len(component)  # cardinality

                if card == 1:  # skip components of size 1
                    continue

                if card < self.vertex_card:
                    self.vertex_card: int = card

                euler: EulerianDiffusion = \
                    EulerianDiffusion(self.graph, component, self.vertex_card)
                euler.diffuse(writer)

                pbar.update()
            pbar.close()


class SubGraphSequencesCSV:
    """
    Like SubGraphSequences but uses h5 files rather than memory.
    """
    def __init__(self, graph: UndirectedGraphCSV, vertex_card: int):
        self.graph: UndirectedGraphCSV = graph
        self.vertex_card: int = vertex_card  # number of nodes per sample

    def extract_components(self, component_file: str) -> List[Set[int]]:
        with jsonlines.open(component_file) as reader:
            components: List[Set[int]] = [set(obj) for obj in reader]
            sizes: List[int] = [len(c) for c in components]
            order: List[int] = np.argsort(sizes)[::-1].tolist()
            components: List[Set[int]] = [components[i] for i in order]

        return components

    def make_subgraph_from_component(self, component: Set[int]) -> UndirectedGraphCSV:
        pass

    def get_sequences(self, component_file: str, out_file: str):
        components: List[int] = self.extract_components(component_file)
        num_components: int = len(components)

        pbar = tqdm(total=num_components)
        with jsonlines.open(out_file, mode='w') as writer:
            for i in range(num_components):
                component_i: Set[int] = components[i]
                card: int = len(component_i)  # cardinality

                if card == 1:  # skip components of size 1
                    continue

                if card < self.vertex_card:
                    self.vertex_card: int = card

                euler: EulerianDiffusion = \
                    EulerianDiffusion(self.graph, component_i, self.vertex_card)
                euler.diffuse(writer)

                pbar.update()
            pbar.close()

    def get_count(self):
        return len(self.graph) + 1
