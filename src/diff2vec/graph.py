"""
NetworkX does not run within 10 hours. We may need to make our own lightweight 
Graph library. 
"""
from collections import defaultdict
from typing import List, Set, Tuple, Dict


class UndirectedGraph:
    """
    Nodes must be integers! We do not assume contiguous nodes.
    """

    def __init__(self):
        self._nodes: Set[int] = {}
        self._edges: List[int] = []

    def add_edge(self, node_a: int, node_b: int):
        self._edges[node_a].append(node_b)
        self._edges[node_b].append(node_a)

    def add_nodes_from(self, nodes: List[int]):
        nodes: Set[int] = set(nodes)
        self._nodes = self._nodes.union(nodes)

    def add_edges_from(self, edges: List[Tuple[int]]):
        for node_a, node_b in edges:
            self.add_edge(node_a, node_b)

    def _dfs(self, path: List[int], node: int, visited: Dict[int, bool]) -> List[int]:
        visited[node] = True  # mark current vertex as visited
        path.append(node)     # add vertex to path

        # repeat for all vertices adjacent to current vertex
        for v in self._edges[node]:
            if not visited[v]:
                path: List[int] = self._dfs(path, v, visited)

        return path

    def connected_components(self):
        visited: Dict[int, bool] = defaultdict(False)
        components: List[Set[int]] = []

        for node in self._nodes:
            if not visited[node]:
                component: List[int] = self._dfs([], node, visited)
                component: Set[int] = set(component)
                components.append(component)

        return components

    def subgraph(self, component: Set[int]):
        """
        Produce a UndirectedGraph instance with only the nodes in the component.
        Any outward edges are removed. We preserve the names in original graph.
        """
        subgraph: UndirectedGraph = UndirectedGraph()
        subgraph.add_nodes_from(component)

    def __len__(self):
        return len(self._nodes)
