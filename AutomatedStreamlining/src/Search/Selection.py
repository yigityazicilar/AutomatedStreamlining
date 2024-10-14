from typing import Any, List, Set, Dict
import math
from Search.Lattice import Lattice
import Util


class UCTSelection:
    exploration_constant: float = 0.1

    def select(
        self, lattice: Lattice, current_combination: Set[str], adjacent_nodes: List[str]
    ) -> str:
        uct_values = self.uct_values(lattice, current_combination, adjacent_nodes)
        return sorted(uct_values.keys(), key=lambda x: uct_values[x], reverse=True)[0]

    def uct_values(
        self, lattice: Lattice, current_combination: Set[str], adjacent_nodes: List[str]
    ) -> Dict[str, float]:
        uct_values: Dict[str, float] = {}
        combination_str_repr: str = Util.get_streamliner_repr_from_set(
            current_combination
        )
        parent_node_attributes: Dict[str, Any] = lattice.get_graph().nodes[
            combination_str_repr
        ]

        for node in adjacent_nodes:
            current_combination.add(node)
            streamliner_combo: str = Util.get_streamliner_repr_from_set(
                current_combination
            )
            cur_attributes: Dict[str, Any] = lattice.get_graph().nodes[
                streamliner_combo
            ]

            # print(f"Current Attributes: {cur_attributes}")

            if cur_attributes["visited_count"] > 0:
                uct_values[node] = (
                    cur_attributes["score"] / cur_attributes["visited_count"]
                ) + self.exploration_constant * math.sqrt(
                    math.log(parent_node_attributes["visited_count"])
                    / cur_attributes["visited_count"]
                )
            else:
                uct_values[node] = float("inf")

            current_combination.remove(node)

        return uct_values
