"""
Eulerian Diffusion.
"""
import os, random
import numpy as np
import json, jsonlines
from tqdm import tqdm
import networkx as nx
from typing import List, Dict, Set
from src.diff2vec.graph import UndirectedGraph


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
        graph: UndirectedGraph,
        component: Set[int],
        cover_size: int):
        self.graph: UndirectedGraph = graph
        self.component: Set[int] = component
        self.cover_size: int = cover_size

    def _diffuse(self, node: int) -> List[int]:
        """
        Generate diffusion tree from source node.
        """
        infected: List[int] = [node] 

        subgraph: nx.DiGraph = nx.DiGraph()  # subgraphs we assume are small enough for nx
        subgraph.add_node(node)  # start with such this node
        counter: int = 1

        while counter < self.cover_size:
            w: int = random.sample(infected, 1)[0]
            neighbors: List[int] = self.graph._edges[w]  

            if len(neighbors) == 0:  # nothing to do!
                break

            u: int = random.sample(neighbors, 1)[0]
            
            if u not in infected:
                counter += 1
                infected.append(u)
                # double graph
                subgraph.add_edges_from([(u, w), (w, u)])

                if counter == self.cover_size:
                    break

        euler: List[int] = [int(u) for u, _ in nx.eulerian_circuit(subgraph, node)]
        return euler

    def diffuse(self, writer: jsonlines.Writer) -> Dict[int, List[int]]:
        pbar = tqdm(total=len(self.component))
        for node in self.component:
            seq: List[int] = self._diffuse(node)
            writer.write(seq)
            pbar.update()
        pbar.close()


class SubGraphSequences:
    """
    Algorithm 2 in https://arxiv.org/pdf/2001.07463.pdf.

    Separate the original graph and run diffusion on each node 
    in subgraph.
    """
    def __init__(self, graph: UndirectedGraph, vertex_card: int):
        self.graph: UndirectedGraph = graph
        self.vertex_card: int = vertex_card  # number of nodes per sample
    
    def extract_components(self, component_file: str) -> List[Set[int]]:
        with jsonlines.open(component_file) as reader:
            components: List[Set[int]] = [set(obj) for obj in reader]
            sizes: List[int] = [len(c) for c in components]
            order: List[int] = np.argsort(sizes)[::-1].tolist()
            components: List[Set[int]] = [components[i] for i in order]

        return components

    def get_sequences(self, components_file: str, out_file: str):
        print('loading connected components...')
        components: List[Set[int]] = self.extract_components(components_file)

        with jsonlines.open(out_file, mode='w') as writer:
            for c, component in enumerate(components):
                card: int = len(component)  # cardinality

                if card == 1:  # skip components of size 1
                    continue

                if card < self.vertex_card:
                    self.vertex_card: int = card

                print(f'Component: ({c+1}/{len(components)})')
                euler: EulerianDiffusion = \
                    EulerianDiffusion(self.graph, component, self.vertex_card)
                euler.diffuse(writer)
