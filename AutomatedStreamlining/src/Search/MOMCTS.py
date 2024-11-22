import copy
import glob
from itertools import chain, combinations
from pathlib import Path
import threading
import time
from turtle import pos

from Toolchain.SolverFactory import get_solver
from Search.StreamlinerModelStats import StreamlinerModelStats
from Portfolio.HydraEval import HydraEval
from Toolchain.Conjure import Conjure
from Toolchain.InstanceStats import InstanceStats, translate_to_instance_stats
from Search.StreamlinerState import StreamlinerState
from SingleModelStreamlinerEvaluation import SingleModelStreamlinerEvaluation

from Search.Lattice import Lattice
import logging, sys, traceback
import random
from typing import Any, List, Set, Tuple, Dict
import pandas as pd
from functools import partial

from Search.Selection import UCTSelection
from Util import get_streamliner_repr_from_set, unwrap
import concurrent.futures
from concurrent.futures import Future


class MOMCTS:

    def __init__(
        self,
        essence_spec: Path,
        training_results: pd.DataFrame,
        eval: HydraEval,
        conf: Dict[str, Any],
        streamliner_model_stats: StreamlinerModelStats,
    ):
        self.essence_spec = essence_spec
        self.working_directory = unwrap(conf.get("working_directory"))
        self.instance_dir = unwrap(conf.get("instance_directory"))
        self.training_results = training_results
        self.eval = eval
        # Generate the set of streamliners for the essence spec
        self.conjure: Conjure = Conjure()
        if res := self.conjure.generate_streamliners(essence_spec):
            stdout, _ = res
            # Generate the streamliner state
            self._streamliner_state = StreamlinerState(
                stdout, conf["streamliners_to_use"]
            )
        else:
            logging.error(f"No streamliners were generated")
            sys.exit(1)
        self._lattice = Lattice()
        self.conf = conf
        self.streamliner_model_stats = streamliner_model_stats
        self.selection_class = UCTSelection()
        self.training_instances: Set[Path] = set(
            [
                Path(instance)
                for instance in glob.glob(str(self.instance_dir / "*.param"))
            ]
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=conf["executor"]["num_cores"]
        )
        self.eval_executor = concurrent.futures.ThreadPoolExecutor()
        self.event = threading.Event()
        self._random = random.Random(42 * 2)


        # if not self.streamliner_model_stats.results().empty:
        #     self._simulate_existing_streamliners()

    # * This threading implementation is an improvement over the initial implementation.
    # * It allows for the simulation of multiple streamliners at the same time without having to wait for all instances to finish.
    # * Will need to be removed to compare against the initial implementation.
    def search(self, portfolio_name: Path) -> None:
        iteration = 0
        maximum_iteration = unwrap(self.conf.get("mcts")).get("num_iterations")
        streamliners_being_run: set[
            tuple[str, concurrent.futures.Future[tuple[Dict[str, InstanceStats], bool]]]
        ] = set()
        thread_count = self.conf["executor"]["num_cores"]
        while not self.event.is_set():
            if len(streamliners_being_run) > 0:
                to_remove: Set[
                    Tuple[str, Future[Tuple[Dict[str, InstanceStats], bool]]]
                ] = set()
                # Check if one is finished and evaluate it. Increment iteration count.
                for streamliner, future in streamliners_being_run:
                    if future.done():
                        try:
                            results, cached = future.result()
                            logging.info(
                                f"Simulation completed for streamliner {streamliner}"
                            )
                        except Exception as _:
                            self._streamliner_state.add_invalid_combination(
                                set(streamliner.split("-"))
                            )
                            logging.error(traceback.format_exc())
                            logging.error(
                                f"Simulation failed for streamliner {streamliner}"
                            )
                            to_remove.add((streamliner, future))
                            continue

                        to_remove.add((streamliner, future))

                        back_prop_value = self.eval.eval_streamliner(
                            streamliner, results, self.training_results
                        )

                        self.eval.save_portfolio_name(
                            portfolio_name.parent
                            / f"{portfolio_name.stem}Iteration{iteration:03d}.json"
                        )

                        self.backprop(streamliner, back_prop_value)

                        if cached:
                            logging.info(f"Streamliner {streamliner} has been cached")
                        else:
                            iteration += 1
                            self._lattice.save_graph(
                                self.working_directory / f"lattice.gexf"
                            )
                            logging.info(
                                f"Iteration {iteration} out of {maximum_iteration}"
                            )

                for remove in to_remove:
                    streamliners_being_run.remove(remove)

            logging.debug(f"Current queue size {self.executor._work_queue.qsize()}")

            if (
                self.executor._work_queue.qsize() <= thread_count
                and iteration < maximum_iteration
            ):
                current_combination, possible_adjacent_streamliners = self.selection()
                if len(current_combination) == 0 and len(possible_adjacent_streamliners) == 0:
                    logging.info("No expandable nodes remain. Waiting for the running ones to finish.")
                    time.sleep(1)
                    continue

                new_combination_added: str = self.expansion(
                    current_combination, list(possible_adjacent_streamliners)
                )

                if new_combination_added in set([s for s, _ in streamliners_being_run]):
                    logging.info(f"Combination {new_combination_added} is already running.")
                    time.sleep(1)
                    continue

                logging.info(f"Adding streamliner {new_combination_added} to the queue")

                simulation_future = self.eval_executor.submit(
                    self.simulation, new_combination_added
                )
                streamliners_being_run.add((new_combination_added, simulation_future))

            if iteration >= maximum_iteration and len(streamliners_being_run) == 0:
                return

            time.sleep(1)

    def selection(self) -> Tuple[Set[str], Set[str]]:
        logging.debug("------SELECTION START-----")
        current_combination: Set[str] = set()
        path_taken: List[str] = []  # Track the path we've taken

        while True:
            logging.info(f"\nSelection iteration:")
            logging.info(f"Current combination: {current_combination}")
            logging.info(f"Current path: {path_taken}")
            
            # Get streamliners that we can combine with our current combination
            possible_adjacent_combinations: Set[str] = (
                self._streamliner_state.get_possible_adjacent_streamliners(
                    current_combination
                )
            )
            logging.info(f"Possible adjacent combinations: {possible_adjacent_combinations}")

            # If no possible adjacent combinations and not at root, backtrack
            if len(possible_adjacent_combinations) == 0:
                logging.info("No possible adjacent combinations found")
                if len(current_combination) == 0:
                    logging.info("At root with no moves, ending selection")
                    return set(), set()
                
                # Store this as a leaf node
                leaf_node = self._streamliner_state.get_streamliner_repr_from_set(
                    current_combination
                )
                logging.info(f"Marking node {leaf_node} as leaf node")
                
                if self._lattice.has_node(leaf_node):
                    self._lattice.get_graph().nodes[leaf_node]['is_leaf'] = True
                    logging.info(f"Node {leaf_node} marked as leaf in lattice")
                else:
                    logging.warning(f"Node {leaf_node} not found in lattice")
                
                # Backtrack by removing last added streamliner
                if path_taken:
                    last_added = path_taken.pop()
                    logging.info(f"Backtracking: removing {last_added} from combination")
                    current_combination.remove(last_added)
                    logging.info(f"After backtrack - combination: {current_combination}, path: {path_taken}")
                else:
                    logging.warning("Attempting to backtrack but path_taken is empty")
                continue

            logging.info("Calculating UCT values")
            uct_values = self.selection_class.uct_values(
                self._lattice, current_combination, possible_adjacent_combinations
            )
            uct_values_sorted = sorted(
                uct_values.items(), key=lambda x: x[1], reverse=True
            )
            logging.info(f"UCT values (sorted): {uct_values_sorted}")

            unexpanded_nodes: Set[str] = set()
            for node, score in uct_values_sorted:
                if score == float("inf"):
                    temp_combination = current_combination | {node}
                    temp_str = self._streamliner_state.get_streamliner_repr_from_set(
                        temp_combination
                    )
                    logging.info(f"Checking if {temp_str} is a leaf node")
                    
                    is_in_lattice = self._lattice.has_node(temp_str)
                    if is_in_lattice:
                        is_leaf = self._lattice.get_graph().nodes[temp_str]['is_leaf']
                        logging.debug(f"Node {temp_str} in lattice: {is_in_lattice}, is_leaf: {is_leaf}")
                        if not is_leaf:
                            unexpanded_nodes.add(node)
                            logging.info(f"Added {node} to unexpanded nodes")
                    else:
                        unexpanded_nodes.add(node)
                        logging.info(f"Added {node} to unexpanded nodes (not in lattice)")

            logging.info(f"Unexpanded nodes found: {unexpanded_nodes}")
            if len(unexpanded_nodes) > 0:
                logging.info(f"Returning current_combination {current_combination} with unexpanded nodes {unexpanded_nodes}")
                return current_combination, unexpanded_nodes

            # Get next node, avoiding known leaf paths
            logging.info("Finding valid moves")
            valid_moves = []
            for node, score in uct_values_sorted:
                temp_combination = current_combination | {node}
                temp_str = self._streamliner_state.get_streamliner_repr_from_set(
                    temp_combination
                )
                
                if self._lattice.has_node(temp_str):
                    is_leaf = self._lattice.get_graph().nodes[temp_str]['is_leaf']
                    if not is_leaf:
                        valid_moves.append(node)
                        logging.info(f"Added {node} to valid moves")
                    else:
                        logging.info(f"Skipping {node} as it leads to leaf node")
                else:
                    valid_moves.append(node)
                    logging.info(f"Added {node} to valid moves (not in lattice)")
            
            logging.info(f"Valid moves found: {valid_moves}")
            if not valid_moves:
                logging.info("No valid moves found, attempting to backtrack")
                if len(current_combination) == 0:
                    logging.info("At root with no valid moves, ending selection")
                    return set(), set()
                if path_taken:
                    last_added = path_taken.pop()
                    logging.info(f"Backtracking: removing {last_added}")
                    current_combination.remove(last_added)
                    logging.info(f"After backtrack - combination: {current_combination}, path: {path_taken}")
                else:
                    logging.warning("Attempting to backtrack but path_taken is empty")
                continue

            highest_score_node = valid_moves[0]
            logging.info(f"Selected move: {highest_score_node}")
            current_combination.add(highest_score_node)
            path_taken.append(highest_score_node)
            logging.info(f"Updated state - combination: {current_combination}, path: {path_taken}")

    def expansion(
        self, current_node_combination: Set[str], possible_adjacent_nodes: List[str]
    ) -> str:
        logging.debug("------EXPANSION------")
        new_candidate: str = possible_adjacent_nodes[
            self._random.randint(0, len(possible_adjacent_nodes) - 1)
        ]

        direct_parent_combination_str = (
            self._streamliner_state.get_streamliner_repr_from_set(
                current_node_combination
            )
        )

        new_node_combination = copy.deepcopy(current_node_combination)
        new_node_combination.add(new_candidate)

        new_streamliner_combo_str = (
            self._streamliner_state.get_streamliner_repr_from_set(new_node_combination)
        )

        # Add the new streamliner combination into the lattice
        self._lattice.add_node(new_streamliner_combo_str)

        # Add an edge between the selected node and the newly expanded node
        self._lattice.add_edge(direct_parent_combination_str, new_streamliner_combo_str)

        # * INFO: This would be the lattice way of doing it however we are using a cache to remember the results so we can do it as a tree.
        # * Add an edge between the selected node and the newly expanded node and other possible parent nodes if they exists in the lattice
        # for comb in combinations(new_node_combination, len(new_node_combination) - 1):
        #     parent_node_combination_str = (
        #         self._streamliner_state.get_streamliner_repr_from_set(set(comb))
        #     )
        #     if self._lattice.has_node(parent_node_combination_str):
        #         self._lattice.add_edge(
        #             parent_node_combination_str, new_streamliner_combo
        #         )

        return new_streamliner_combo_str

    def simulation(self, new_combination: str) -> Tuple[Dict[str, InstanceStats], bool]:
        logging.debug("------SIMULATION------")
        streamliner_results_df = self.streamliner_model_stats.results()

        instances_left_to_eval: Set[Path] = set()
        instances_in_results: Set[str] = set(
            streamliner_results_df[
                streamliner_results_df["Streamliner"] == new_combination
            ]["Instance"]
        )

        for instance in self.training_instances:
            if instance.name not in instances_in_results:
                instances_left_to_eval.add(instance)

        # Check to see if we have already encountered this streamliner before
        if len(instances_left_to_eval) == 0:
            logging.info(
                f"{new_combination} has already been seen. Using cached results"
            )
            results = streamliner_results_df[
                streamliner_results_df["Streamliner"] == new_combination
            ]
            base_results = {}
            for _, row in results.iterrows():
                base_results[row["Instance"]] = translate_to_instance_stats(row)  # type: ignore

            if len(results) >= len(self.training_results["Instance"]):
                return base_results, True

        generated_models = self.conjure.generate_streamlined_models(
            self.essence_spec,
            new_combination,
            output_dir=self.working_directory / "conjure-output" / new_combination,
        )
        if len(generated_models) == 1:
            instances_to_run: Set[Path] = self._get_instances_to_run(
                new_combination, streamliner_results_df
            )
            # Instances that have not been run yet. Allows for midway restarts
            instances_to_run.intersection_update(instances_left_to_eval)

            # instances_to_run = instances_left_to_eval

            if len(instances_to_run) == 0:
                return {}, False

            streamlinerEval = SingleModelStreamlinerEvaluation(
                model=generated_models[0],
                instances=instances_to_run,
                stats=self.training_results,
                solver=get_solver(unwrap(self.conf.get("solver"))),
                executor=self.executor,
                time=lambda x: x * 1.05,
                event=self.event,
                streamliner=new_combination,
            )
            callback = partial(self.streamliner_model_stats.callback, new_combination)
            # We now need to parse these results into some format that we can use as a reference point
            base_results = streamlinerEval.execute(callback=callback)

            return base_results, False
        else:
            raise ValueError("More than one streamlined model was generated.")

    def backprop(self, streamliner_combo: str, back_prop_value: int):
        logging.debug("------BACKPROP-----")

        """
        Back prop up the lattice structure to the root node 
        """

        node_attributes = self._lattice.get_graph().nodes[streamliner_combo]
        node_attributes["visited_count"] = node_attributes["visited_count"] + 1
        node_attributes["score"] += back_prop_value
        logging.debug(f"Node attributes: {node_attributes}")

        predecessor_nodes = set(
            self._lattice.get_graph().predecessors(streamliner_combo)
        )
        logging.debug(f"Predecessor Nodes: {predecessor_nodes}")
        for node in predecessor_nodes:
            self.backprop(node, back_prop_value)

    # * INFO: This function is used to get the instances that need to be run for a given streamliner combination
    # * It is an improvement over the initial implementation.
    def _get_instances_to_run(
        self, streamliner_comb: str, streamliner_results_df: pd.DataFrame
    ) -> Set[Path]:
        # ! Use combinations to get all combinations and if they are in the lattice intersect them.
        combination_set = set(streamliner_comb.split("-"))
        if len(combination_set) == 1:
            return set(self.training_instances)

        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

        all_combinations: List[Tuple[str, ...]] = list(powerset(combination_set))

        def get_valid_instances_for_streamliner(streamliner: str) -> Set[str]:
            """Helper function to get valid instances for a given streamliner."""
            mask = (streamliner_results_df["Streamliner"] == streamliner) & (
                streamliner_results_df["Satisfiable"]
                | streamliner_results_df["TimeOut"]
            )
            return set(streamliner_results_df.loc[mask]["Instance"])

        first_streamliner = get_streamliner_repr_from_set(set(all_combinations[0]))
        instances_to_run_str = get_valid_instances_for_streamliner(first_streamliner)

        # Intersect with remaining streamliners
        for streamliner in list(all_combinations)[1:]:
            instances_to_run_str.intersection_update(
                get_valid_instances_for_streamliner(
                    get_streamliner_repr_from_set(set(streamliner))
                )
            )

        instances_to_run = set()
        for instance in instances_to_run_str:
            instances_to_run.add(
                next(
                    instance_path
                    for instance_path in self.training_instances
                    if instance == instance_path.name
                )
            )

        return instances_to_run

    # * If the run is stopped midway this will simulate the streamliners that have already been run.
    # ? However, this was not needed as we did not stop the run midway.
    def _simulate_existing_streamliners(self):
        tested_streamliners_df = self.streamliner_model_stats.results()
        tested_streamliners = self.streamliner_model_stats.results()[
            "Streamliner"
        ].unique()
        for simulated_streamliner in tested_streamliners:
            results = tested_streamliners_df[
                tested_streamliners_df["Streamliner"] == simulated_streamliner
            ]
            self._lattice.add_node(simulated_streamliner)
            streamliner_set = set(simulated_streamliner.split("-"))
            for comb in combinations(streamliner_set, len(streamliner_set) - 1):
                parent_node_combination_str = (
                    self._streamliner_state.get_streamliner_repr_from_set(set(comb))
                )
                if self._lattice.has_node(parent_node_combination_str):
                    self._lattice.add_edge(
                        parent_node_combination_str, simulated_streamliner
                    )

            base_results = {}
            for _, row in results.iterrows():
                base_results[row["Instance"]] = translate_to_instance_stats(row)

            back_prop_value = self.eval.eval_streamliner(
                simulated_streamliner, base_results, self.training_results
            )
            self.backprop(simulated_streamliner, back_prop_value)
