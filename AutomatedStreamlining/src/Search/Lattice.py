from pathlib import Path
from typing import Set
import logging
from networkx import DiGraph, write_gexf


class Lattice:
    def __init__(self):
        self.__graph: DiGraph = DiGraph()
        self.__created = 0
        # Add the root node of the lattice (the unstreamlined model)
        self.add_node("")

    def streamliner_combo_str_repr(self, streamliner_combo: Set[str]) -> str:
        return ",".join(
            sorted(
                [candidate_streamliner for candidate_streamliner in streamliner_combo]
            )
        )

    def add_node(self, node_combination: str) -> None:
        logging.debug(f"Adding node {node_combination}")
        self.__graph.add_node(
            node_combination,
            visited_count=0,
            score=0,
            is_leaf=False,
            created=self.__created,
        )
        self.__created += 1

    def add_edge(self, current_combo_str: str, streamliner_combo: str) -> None:
        logging.debug(
            f"Adding edge between {0 if current_combo_str == '' else current_combo_str} and {streamliner_combo}"
        )
        self.__graph.add_edge(current_combo_str, streamliner_combo)

    def get_graph(self) -> DiGraph:
        return self.__graph

    def save_graph(self, filename: Path) -> None:
        write_gexf(self.__graph, filename)

    def has_node(self, node_combination: str) -> bool:
        return self.__graph.has_node(node_combination)
