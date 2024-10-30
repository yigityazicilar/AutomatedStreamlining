import copy
import glob
from itertools import combinations
import os
from pathlib import Path
import signal
import time

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
from Util import unwrap
import concurrent.futures


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
        self.training_instances: set[str] = set(
            [
                instance.split("/")[-1]
                for instance in glob.glob(f"{self.instance_dir}/*.param")
            ]
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=conf["executor"]["num_cores"])
        self.eval_executor = concurrent.futures.ThreadPoolExecutor()
        # if not self.streamliner_model_stats.results().empty:
        #     self._simulate_existing_streamliners()

    #* This threading implementation is an improvement over the initial implementation.
    #* It allows for the simulation of multiple streamliners at the same time without having to wait for all instances to finish.
    #* Will need to be removed to compare against the initial implementation.
    def search(self, portfolio_name: str | None = None) -> None:
        iteration = 0
        streamliner_being_run: set[tuple[str, concurrent.futures.Future[tuple[Dict[str, InstanceStats], bool]]]] = set()
        thread_count = self.conf["executor"]["num_cores"]
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        while True:
            if len(streamliner_being_run) > 0:
                to_remove = set()
                # Check if one is finished and evaluate it. Increment iteration count.
                for streamliner, future in streamliner_being_run:
                    logging.info(f"Checking if streamliner {streamliner} finished running.")
                    if future.done():
                        try:
                            results, cached = future.result()
                        except Exception as _:
                            self._streamliner_state.add_invalid_combination(current_combination)
                            logging.error(traceback.format_exc())
                            logging.error("Simulation failed")
                            to_remove.add((streamliner, future))
                            continue

                        to_remove.add((streamliner, future))

                        back_prop_value = self.eval.eval_streamliner(
                            streamliner, results, self.training_results
                        )

                        if portfolio_name is not None:
                            portfolio_name = portfolio_name.split(".")[0]
                            self.eval.save_portfolio_name(
                                f"{portfolio_name}Iteration{iteration}.json"
                            )
                        else:
                            self.eval.save_portfolio()

                        self.backprop(streamliner, back_prop_value)

                        if cached:
                            logging.info("Cached result")
                        else:
                            iteration += 1
                            logging.info(
                                f"Iteration {iteration} out of {unwrap(self.conf.get('mcts')).get('num_iterations')}"
                            )
                            
                for remove in to_remove:
                    streamliner_being_run.remove(remove)
         
            logging.info(f"Current queue size {self.executor._work_queue.qsize()}")
            if self.executor._work_queue.qsize() <= thread_count:
                logging.info(f"Adding new streamliners to the queue")
                current_combination, possible_adjacent_streamliners = self.selection()
                new_combination_added: str = self.expansion(
                    current_combination, list(possible_adjacent_streamliners)
                )
                
                if new_combination_added in streamliner_being_run:
                    continue

                simulation_future = self.eval_executor.submit(self.simulation, new_combination_added)
                streamliner_being_run.add((new_combination_added, simulation_future))
                    
            if iteration >= unwrap(self.conf.get("mcts")).get("num_iterations"):
                return

            time.sleep(5)

    def selection(self) -> Tuple[Set[str], Set[str]]:
        logging.debug("------SELECTION-----")
        current_combination: Set[str] = set()

        while True:
            # Get streamliners that we can combine with our current combination
            possible_adjacent_combinations: Set[str] = (
                self._streamliner_state.get_possible_adjacent_streamliners(
                    current_combination
                )
            )

            combination_str_repr: str = (
                self._streamliner_state.get_streamliner_repr_from_set(
                    current_combination
                )
            )

            # Find the adjacent nodes in the Lattice to our current combination
            adjacent_nodes: List[str] = list(
                self._lattice.get_graph().neighbors(combination_str_repr)
            )

            # Calculate if all possible children exist in the Lattice
            set_diff = set(possible_adjacent_combinations) - set(adjacent_nodes)

            # If not all children have been created, stop and expand this node
            if len(set_diff) > 0:
                logging.debug(
                    f"Not all children have been created. Returning {current_combination}"
                )
                return current_combination, set_diff
            # Else move down the Lattice and continue to select
            else:
                node = self.selection_class.select(
                    self._lattice, current_combination, adjacent_nodes
                )
                current_combination.add(node)

    def expansion(
        self, current_node_combination: Set[str], possible_adjacent_nodes: List[str]
    ) -> str:
        logging.debug("------EXPANSION------")
        new_candidate: str = possible_adjacent_nodes[
            random.randint(0, len(possible_adjacent_nodes) - 1)
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


        #* INFO: This would be the lattice way of doing it however we are using a cache to remember the results so we can do it as a tree.
        #* Add an edge between the selected node and the newly expanded node and other possible parent nodes if they exists in the lattice
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
        logging.info(f"New combo {new_combination}")

        instances_left_to_eval = self.training_instances - set(
            streamliner_results_df[
                streamliner_results_df["Streamliner"] == new_combination
            ]["Instance"]
        )

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
                base_results[row["Instance"]] = translate_to_instance_stats(row)

            if len(results) >= len(self.training_results["Instance"]):
                return base_results, True

        generated_models = self.conjure.generate_streamlined_models(
            self.essence_spec,
            new_combination,
            output_dir=os.path.join(self.working_directory, "conjure-output"),
        )
        if len(generated_models) == 1:
            instances_to_run: Set[str] = self._get_instances_to_run(new_combination, streamliner_results_df)
            # Instances that have not been run yet. Allows for midway restarts
            instances_to_run.intersection_update(instances_left_to_eval)

            if len(instances_to_run) == 0:
                return {}, False

            streamlinerEval = SingleModelStreamlinerEvaluation(
                generated_models[0],
                self.working_directory,
                self.instance_dir,
                instances_to_run,
                self.training_results,
                get_solver(unwrap(self.conf.get("solver"))),
                self.executor,
                None,
                lambda x: x * 1.5,
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

    #* INFO: This function is used to get the instances that need to be run for a given streamliner combination
    #* It is an improvement over the initial implementation.
    def _get_instances_to_run(self, streamliner_comb: str, streamliner_results_df: pd.DataFrame) -> Set[str]:
        combination_set = set(streamliner_comb.split("-"))
        if len(combination_set) == 1:
            return set(self.training_instances)
        
        def get_valid_instances_for_streamliner(streamliner):
            """Helper function to get valid instances for a given streamliner."""
            mask = (
                (streamliner_results_df["Streamliner"] == streamliner) &
                (streamliner_results_df["Satisfiable"] | streamliner_results_df["TimeOut"])
            )
            return set(streamliner_results_df.loc[mask]["Instance"])

        first_streamliner = next(iter(combination_set))
        instances_to_run = get_valid_instances_for_streamliner(first_streamliner)
        
        # Intersect with remaining streamliners
        for streamliner in list(combination_set)[1:]:
            instances_to_run.intersection_update(
                get_valid_instances_for_streamliner(streamliner)
            )
        
        return instances_to_run
    
    def signal_handler(self, sig, frame):
        logging.info(f"Caught signal shutting down process pools...")
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.eval_executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(0)

    #* If the run is stopped midway this will simulate the streamliners that have already been run.
    #? However, this was not needed as we did not stop the run midway.
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
            self.eval.save_portfolio()

            self.backprop(simulated_streamliner, back_prop_value)