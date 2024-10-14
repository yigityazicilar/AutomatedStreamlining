import json
import logging
from typing import AnyStr, Set, List, FrozenSet, Dict, Any

from StreamlinerSelection import StreamlinerSelection

selected_streamliner_groups = ["MatrixByRowBucket-0", "MatrixByRowBucket-9"]
removed_streamliner_groups = ["MatrixBy"]


class StreamlinerState:
    def __init__(self, streamliner_output: AnyStr, streamliners_to_use: StreamlinerSelection):
        self.streamliner_json: Dict[str, Any] = json.loads(streamliner_output)
        temp_streamliner_dict: Dict[str, Any] = {}
        
        if streamliners_to_use == StreamlinerSelection.FINE_FILTERED:
            for k, v in self.streamliner_json.items():
                for s in selected_streamliner_groups:
                    if s in v["groups"]:
                        v["groups"].remove(s)
                        temp_streamliner_dict[k] = v
        elif streamliners_to_use == StreamlinerSelection.COARSE:
            for k, v in self.streamliner_json.items():
                for r in removed_streamliner_groups:
                    filtered = False
                    for g in v["groups"]:
                        if r in g:
                            filtered = True
                            break
                    if not filtered:
                        temp_streamliner_dict[k] = v
        else:
           for k, v in self.streamliner_json.items():
                for r in removed_streamliner_groups:
                    filtered = False
                    for g in v["groups"]:
                        if r in g:
                            filtered = True
                            break
                    if filtered:
                        temp_streamliner_dict[k] = v 
                        
        self.streamliner_json = temp_streamliner_dict
        logging.info(f"Number of candidate streamliners: {len(self.streamliner_json)}")
        self.init_groups(self.streamliner_json)
        self.invalid_combinations: Set[FrozenSet[str]] = set()

    def init_groups(self, streamliner_json: Dict[str, Any]) -> None:
        self.groups = {}
        for val in streamliner_json:
            for group in streamliner_json[val]["groups"]:
                if group in self.groups:
                    self.groups[group].add(val)
                else:
                    self.groups[group] = set(val)

    def get_candidate_streamliners(self) -> Set[str]:
        return set(self.streamliner_json.keys())

    def get_possible_adjacent_streamliners(
        self, streamliner_combination: Set[str]
    ) -> Set[str]:
        possible_adjacent_streamliners: Set[str] = (
            self.get_candidate_streamliners() - streamliner_combination
        )
        to_remove: Set[str] = set()
        for streamliner in streamliner_combination:
            groups: List[str] = self.streamliner_json[streamliner]["groups"]

            for candidate_streamliner in possible_adjacent_streamliners:
                if not set(groups) - set(
                    self.streamliner_json[candidate_streamliner]["groups"]
                ):
                    to_remove.add(candidate_streamliner)

                current_set: Set[str] = set.union(
                    set(candidate_streamliner), streamliner_combination
                )
                if set.intersection(current_set, self.invalid_combinations):
                    to_remove.add(candidate_streamliner)

        possible_adjacent_streamliners.difference_update(to_remove)
        return possible_adjacent_streamliners

    def get_possible_adjacent_combinations(
        self, current_streamliner_combination: Set[str]
    ) -> Set[str]:
        adjacent_streamliners: Set[str] = self.get_possible_adjacent_streamliners(
            current_streamliner_combination
        )
        return set(
            [
                self.get_streamliner_repr_from_set(
                    current_streamliner_combination.union(set([streamliner]))
                )
                for streamliner in adjacent_streamliners
            ]
        )

    def get_streamliner_repr_from_set(self, streamliner_combo: Set[str]) -> str:
        return "-".join([str(i) for i in sorted([int(s) for s in streamliner_combo])])

    def add_invalid_combination(self, combination: Set[str]) -> None:
        logging.info("Adding invalid combination")
        self.invalid_combinations.add(frozenset(combination))
