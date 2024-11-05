import json, sys
import logging
from logging import config
from typing import AnyStr, Set, List, FrozenSet, Dict, Any

from StreamlinerSelection import StreamlinerSelection


class StreamlinerState:
    SELECTED_GROUPS = ["MatrixByRowBucket-0", "MatrixByRowBucket-9"]
    SCOPES = ["MatrixByRowBucket", "MatrixByColBucket"]

    def __init__(
        self, streamliner_output: AnyStr, streamliners_to_use: StreamlinerSelection
    ) -> None:
        raw_streamliners = json.loads(streamliner_output)
        scope_streamliners = self._change_to_scope(raw_streamliners)
        self.streamliner_json = self._filter_streamliners(
            scope_streamliners, streamliners_to_use
        )
        logging.info(f"Number of candidate streamliners: {len(self.streamliner_json)}")
        self.invalid_combinations: Set[FrozenSet[str]] = set()

    def _change_to_scope(self, streamliners: Dict[str, Any]) -> Dict[str, Any]:
        for id, config in streamliners.items():
            for group in config["groups"]:
                for change in self.SCOPES:
                    if change in group:
                        config["groups"].remove(group)
                        config["scope"] = group
                        break
                if config.get("scope", None) is None:
                    config["scope"] = "MatrixWide"

            streamliners[id] = config

        return streamliners

    def _filter_streamliners(
        self, streamliners: Dict[str, Any], mode: StreamlinerSelection
    ) -> Dict[str, Any]:
        filtered = {}
        match mode:
            case StreamlinerSelection.FINE_FILTERED:
                for id, config in streamliners.items():
                    if config["scope"] in self.SELECTED_GROUPS:
                        filtered[id] = config
            case StreamlinerSelection.COARSE:
                for id, config in streamliners.items():
                    if config["scope"] == "MatrixWide":
                        filtered[id] = config
            case StreamlinerSelection.FINE:
                for id, config in streamliners.items():
                    if config["scope"] != "MatrixWide":
                        filtered[id] = config
            case StreamlinerSelection.ALL:
                filtered = streamliners
        return filtered

    def get_candidate_streamliners(self) -> Set[str]:
        return set(self.streamliner_json.keys())

    def get_possible_adjacent_streamliners(
        self, streamliner_combination: Set[str]
    ) -> Set[str]:
        candidates: Set[str] = (
            self.get_candidate_streamliners() - streamliner_combination
        )
        invalid_candidates: Set[str] = set()

        for streamliner in streamliner_combination:
            current_scope: str = self.streamliner_json[streamliner]["scope"]
            current_groups = set(self.streamliner_json[streamliner]["groups"])

            for candidate in candidates:
                candidate_groups = set(self.streamliner_json[candidate]["groups"])
                candidate_scope = self.streamliner_json[candidate]["scope"]
                if (
                    len(current_groups.intersection(candidate_groups)) != 0
                    and current_scope == candidate_scope
                ):
                    invalid_candidates.add(candidate)

                # Check if combination would be invalid
                combination: Set[str] = set([candidate]) | streamliner_combination
                if any(
                    invalid_set == combination
                    for invalid_set in self.invalid_combinations
                ):
                    invalid_candidates.add(candidate)

        return candidates - invalid_candidates

    def get_possible_adjacent_combinations(
        self, current_combination: Set[str]
    ) -> Set[str]:
        adjacent_streamliners = self.get_possible_adjacent_streamliners(
            current_combination
        )
        return {
            self.get_streamliner_repr_from_set(current_combination | {streamliner})
            for streamliner in adjacent_streamliners
        }

    def get_streamliner_repr_from_set(self, streamliner_combo: Set[str]) -> str:
        return "-".join(str(i) for i in sorted(int(s) for s in streamliner_combo))

    def add_invalid_combination(self, combination: Set[str]) -> None:
        logging.info("Adding invalid combination")
        self.invalid_combinations.add(frozenset(combination))
