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
        selected_node = sorted(
            uct_values.keys(), key=lambda x: uct_values[x], reverse=True
        )[0]
        logging.debug(
            f"Node {selected_node} selected with UCT value {uct_values[selected_node]}"
        )
        return selected_node

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

        for node in adjacent_nodes:
            updated_combination = current_combination | {
                node
            }  # Create a new combination set
            streamliner_combo: str = Util.get_streamliner_repr_from_set(
                updated_combination
            )
            cur_attributes: Dict[str, Any] = lattice.get_graph().nodes[
                streamliner_combo
            ]

            # Calculate the UCT value
            if cur_attributes["visited_count"] > 0:
                uct_values[node] = (
                    cur_attributes["score"] / cur_attributes["visited_count"]
                ) + self.UCT_EXPLORATION_CONSTANT * math.sqrt(
                    math.log(parent_node_attributes["visited_count"])
                    / cur_attributes["visited_count"]
                )
            else:
                uct_values[node] = float("inf")

        logging.debug(f"UCT values calculated: {uct_values}")
        return uct_values
