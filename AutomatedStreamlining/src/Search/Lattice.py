from typing import Set
import logging
from networkx import DiGraph


class LatticeNode:
    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, LatticeNode)
            and o.streamliner_combination() == self._streamliner_combination
        )

    def __str__(self) -> str:
        return ",".join(sorted([x for x in self._streamliner_combination]))

    """ Return the hash of the sorted individual streamliners"""

    def __hash__(self) -> int:
        return self._streamliner_combination.__hash__()

    def __init__(self, streamliner_combination: Set[str]):
        self.children = {}
        self._streamliner_combination = frozenset(streamliner_combination)

    def streamliner_combination(self):
        return self._streamliner_combination


class Lattice:
    def __init__(self):
        self.__graph: DiGraph = DiGraph()
        # Add the root node of the lattice (the unstreamlined model)
        self.add_node("")

    def streamliner_combo_str_repr(self, streamliner_combo: Set[str]) -> str:
        return ",".join(
            sorted(
                [candidate_streamliner for candidate_streamliner in streamliner_combo]
            )
        )

    def add_node(self, node_combination: str) -> None:
        logging.info(f"Adding node {node_combination}")
        self.__graph.add_node(node_combination, visited_count=0, score=0)

    def add_edge(self, current_combo_str: str, streamliner_combo: str) -> None:
        logging.info(
            f"Adding edge between {0 if current_combo_str == '' else current_combo_str} and {streamliner_combo}"
        )
        self.__graph.add_edge(current_combo_str, streamliner_combo)

    def get_graph(self) -> DiGraph:
        return self.__graph

    def has_node(self, node_combination: str) -> bool:
        return self.__graph.has_node(node_combination)
