"""
NetworkX does not run within 10 hours. We may need to make our own lightweight 
Graph library. 
"""
import os
import sys
import csv
import json
import h5py
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List, Set, Tuple, Dict, Union, Any

from src.utils.utils import to_json


class UndirectedGraph:
    """
    Nodes must be integers! We do not assume contiguous nodes.
    """

    def __init__(self):
        self._nodes: Set[int] = set()
        self._edges: Dict[int, List[int]] =  defaultdict(lambda: [])

    def add_node(self, node: int):
        self._nodes.add(node)

    def add_edge(self, node_a: int, node_b: int):
        self._edges[node_a].append(node_b)
        self._edges[node_b].append(node_a)

    def add_nodes_from(self, nodes: Union[List[int], Set[int]]):
        nodes: Set[int] = set(nodes)
        self._nodes = self._nodes.union(nodes)

    def add_edges_from(self, edges: List[Tuple[int]]):
        pbar = tqdm(total=len(edges))
        for node_a, node_b in edges:
            self.add_edge(node_a, node_b)
            pbar.update()
        pbar.close()

    def _dfs(
        self,
        path: List[int],
        node: int,
        visited: Dict[int, bool]
    ) -> List[int]:
        visited[node] = True  # mark current vertex as visited
        path.append(node)     # add vertex to path

        # repeat for all vertices adjacent to current vertex
        for v in self._edges[node]:
            if not visited[v]:
                path: List[int] = self._dfs(path, v, visited)

        return path

    def has_node(self, node: int) -> bool:
        return node in self._nodes

    def nodes(self) -> Set[int]:
        return self._nodes

    def neighbors(self, node: int) -> Set[int]:
        assert self.has_node(node), "Graph does not contain node"
        # since we always add backwards connections, we can just fetch
        # all the connections frmo this node.
        return set(self._edges[node]) - {node}

    def connected_components(self) -> List[Set[int]]:
        sys.setrecursionlimit(len(self._nodes))
        visited: Dict[int, bool] = defaultdict(lambda: False)
        components: List[Set[int]] = []

        pbar = tqdm(total=len(self._nodes))
        for node in self._nodes:
            if not visited[node]:
                component: List[int] = self._dfs([], node, visited)
                component: Set[int] = set(component)
                components.append(component)

            pbar.update()
        pbar.close()

        return components

    def subgraph(self, component: Set[int]):
        """
        Produce a UndirectedGraph instance with only the nodes in the component.
        We preserve the names in original graph. 
        """
        subgraph: UndirectedGraph = UndirectedGraph()
        subgraph.add_nodes_from(component)

        for node in component:
            links: Set[int] = self.neighbors(node)
            links: Set[int] = links.intersection(component)
            subgraph._edges[node] = list(links) 

        return subgraph

    def is_connected(self) -> bool:
        """
        Method to check if all non-zero degree vertices are connected. It 
        mainly does DFS traversal starting from node with non-zero degree.
        """
        visited: Dict[int, bool] = defaultdict(False)

        # find a vertex with non-zero degree
        for node in self._nodes:
            if len(self._edges[node]) > 1:
                break

        # if no edges, return true
        if node == self.__len__() - 1:
            return True

        # traverse starting from vertex with non-zero degree
        _ = self._dfs([], node, visited)

        # check if all non-zero vertices are visited
        for node in self._nodes:
            if not visited[node] and len(self._edges[node]) > 0:
                return False

        return True

    def is_eulerian(self) -> bool:
        """The function returns one of the following values:
        0 --> If graph is not Eulerian
        1 --> If graph has an Euler path (Semi-Eulerian)
        2 --> If graph has an Euler Circuit (Eulerian)
        """
        if not self.is_connected():
            return 0

        # Count vertices with odd degree
        odd: int = 0
        for node in self._nodes:
            if len(self._edges[node]) % 2 != 0:
                odd += 1

        # If odd count is 2, then semi-eulerian.
        # If odd count is 0, then eulerian.
        # If count is more than 2, then graph is not Eulerian
        # Note that odd count can never be 1 for undirected graph.
        if odd == 0:
            return 2
        elif odd == 1:
            raise Exception('This is not possible!')
        elif odd == 2:
            return 1
        else: 
            return 0

    def to_h5(self, nodes_file, edges_file):
        with h5py.File(edges_file, 'w') as fp:
            pbar = tqdm(total=len(self._nodes))
            for node in self._edges:
                fp.create_dataset(str(node), data = self._edges[node])
                pbar.update()
            pbar.close()

        to_json(self._nodes, nodes_file)

    def to_csv(self, nodes_file, edges_file):
        with open(edges_file, 'a') as fp:
            writer = csv.writer(fp)
            writer.writerow(['nodes', 'edges'])
            pbar = tqdm(total=len(self._nodes))
            for node in self._edges:
                edges: str = json.dump(self._edges[node])
                node: str = str(node)
                writer.writerow([node, edges])
                pbar.update()
            pbar.close()
        to_json(self._nodes, nodes_file)

    def __len__(self) -> int:
        return len(self._nodes)


class UndirectedGraphH5:
    """
    Like UndirectedGraph but memory efficient so only works from an h5 file.
    Nodes must be integers! We do not assume contiguous nodes.
    """
    def __init__(self, nodes_file: str, edges_file: str):
        self._nodes_file: str = nodes_file
        self._edges_file: str = edges_file

        self._nodes: Set[int] = set(json.load(open(nodes_file)))
        self._edges_fp: Any = h5py.File(edges_file, 'r')
        self._component_dir = None

    def has_node(self, node: int) -> bool:
        return node in self._nodes

    def nodes(self) -> Set[int]:
        return self._nodes

    def neighbors(self, node: int) -> Set[int]:
        assert self.has_node(node), "Graph does not contain node"
        return set(self._edges_fp[str(node)]) - {node}  # no self loops

    def connected_components(self, out_dir: str) -> List[Set[int]]:
        sys.setrecursionlimit(len(self._nodes))
        visited: Dict[int, bool] = defaultdict(lambda: False)
        count: int = 0
        sizes: List[int] = []

        pbar = tqdm(total=len(self._nodes))
        for node in self._nodes:
            if not visited[node]:
                component: List[int] = self._dfs([], node, visited)
                component: Set[int] = set(component)
                subgraph: UndirectedGraph = self.subgraph(component)
                sub_nodes_file: str = os.path.join(out_dir, f'component{count}-nodes.json')
                sub_edges_file: str = os.path.join(out_dir, f'component{count}-edges.h5')
                subgraph.to_h5(sub_nodes_file, sub_edges_file)
                sizes[count] = len(component)
                count += 1

            pbar.update()
        pbar.close()

        to_json(sizes, os.path.join(out_dir, 'component-sizes.json'))
        self._component_dir: str = out_dir

    def subgraph(self, component: Set[int]):
        subgraph: UndirectedGraph = UndirectedGraph()
        subgraph.add_nodes_from(component)

        for node in component:
            links: Set[int] = self.neighbors(node)
            links: Set[int] = links.intersection(component)
            subgraph._edges[node] = list(links) 

        return subgraph

    def __len__(self) -> int:
        return len(self._nodes)
