from typing import Any, Set, Dict
import math
from Search.Lattice import Lattice
import Util
import logging


class UCTSelection:
    UCT_EXPLORATION_CONSTANT: float = 0.1

    def select(
        self, lattice: Lattice, current_combination: Set[str], adjacent_nodes: Set[str]
    ) -> str:
        uct_values = self.uct_values(lattice, current_combination, adjacent_nodes)
        return sorted(uct_values.keys(), key=lambda x: uct_values[x], reverse=True)[0]

    def uct_values(
        self, lattice: Lattice, current_combination: Set[str], adjacent_nodes: Set[str]
    ) -> Dict[str, float]:
        uct_values: Dict[str, float] = {}
        combination_str_repr: str = Util.get_streamliner_repr_from_set(
            current_combination
        )
        parent_node_attributes: Dict[str, Any] = lattice.get_graph().nodes[
            combination_str_repr
        ]

        # Work with a copy to avoid modifying the original
        temp_combination = current_combination.copy()

        for node in adjacent_nodes:
            temp_combination.add(node)
            streamliner_combo: str = Util.get_streamliner_repr_from_set(
                temp_combination
            )
            logging.debug(f"Processing node {node} in combination {streamliner_combo}")

            if lattice.get_graph().has_node(streamliner_combo):
                cur_attributes: Dict[str, Any] = lattice.get_graph().nodes[
                    streamliner_combo
                ]

                if cur_attributes["visited_count"] > 0:
                    uct_values[node] = (
                        cur_attributes["score"] / cur_attributes["visited_count"]
                    ) + self.UCT_EXPLORATION_CONSTANT * math.sqrt(
                        math.log(parent_node_attributes["visited_count"])
                        / cur_attributes["visited_count"]
                    )
                else:
                    # ! This section should never be reached but to be safe we will keep it.
                    logging.debug(
                        f"Node {streamliner_combo} has not been visited, setting UCT to inf"
                    )
                    uct_values[node] = float("inf")
            else:
                logging.debug(
                    f"Node {streamliner_combo} not found in graph, setting UCT to inf"
                )
                uct_values[node] = float("inf")

            temp_combination.remove(node)

        return uct_values
