"""
NetworkX does not run within 10 hours. We may need to make our own lightweight 
Graph library. 
"""
import os
import sys
import csv
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from typing import List, Set, Tuple, Dict, Union


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

    def to_csv(self, edges_file: str):
        with open(edges_file, 'a') as fp:
            writer = csv.writer(fp)
            writer.writerow(['nodes', 'edges'])
            pbar = tqdm(total=len(self._nodes))
            max_node: int = max(self._nodes)
            missing: int = 0
            for node in range(max_node): 
                if node in self._edges:
                    edges: List[str] = json.dumps(self._edges[node])
                else:
                    edges: List[str] = json.dumps([])
                    missing += 1
                node: str = str(node)
                writer.writerow([node, edges])
                pbar.update()
            pbar.close()
        print(f'{missing} missing.')

    def __len__(self) -> int:
        return len(self._nodes)


class UndirectedGraphCSV:

    def __init__(self, edges_dir: str):
        _edges_files: List[str] = glob(os.path.join(edges_dir, '*.csv'))
        _edges_files: List[str] = sorted(
            _edges_files, key=lambda x: int(x.split('.')[-2].split('-')[1]))[-1]
        self._size: int = pd.read_csv(_edges_files[-1]).nodes.max()
        self._split_size: int = len(pd.read_csv(_edges_files[0]))
        self._edges_dir: str = edges_dir

    def _dfs(
        self,
        path: List[int],
        node: int,
        visited: Dict[int, bool]
    ) -> List[int]:
        visited[node] = True  # mark current vertex as visited
        path.append(node)     # add vertex to path

        # repeat for all vertices adjacent to current vertex
        for v in self.neighbors(node):
            if not visited[v]:
                path: List[int] = self._dfs(path, v, visited)

        return path

    def neighbors(self, node: int) -> Set[int]:
        index: int = node // self._split_size
        df: pd.DataFrame = pd.read_csv(os.path.join(self._edges_dir, f'edges-{index}.csv'))
        edges: str = df.iloc[node % self._split_size].edges
        edges: List[int] = json.loads(edges)
        return set(edges) - {node}

    def connected_components(self) -> List[Set[int]]:
        sys.setrecursionlimit(self._size)
        visited: Dict[int, bool] = defaultdict(lambda: False)
        sizes: List[int] = []
        components: List[Set[str]] = []

        breakpoint()
        pbar = tqdm(total=self._size)
        for node in range(self._size):
            if not visited[node]:
                component: List[int] = self._dfs([], node, visited)
                component: Set[int] = set(component)
                if len(component) > 1:
                    components.append(component)
                    sizes.append(len(component))

            pbar.update()
        pbar.close()

        return components

    def subgraph(self, component: Set[int]):
        subgraph: UndirectedGraph = UndirectedGraph()
        subgraph.add_nodes_from(component)

        for node in component:
            links: Set[int] = self.neighbors(node)
            links: Set[int] = links.intersection(component)
            subgraph._edges[node] = list(links) 

        return subgraph

    def __len__(self) -> int:
        return self._size
