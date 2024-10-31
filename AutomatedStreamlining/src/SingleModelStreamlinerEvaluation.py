import logging
from pathlib import Path
import threading

import pandas as pd
from Toolchain.Pipeline import Pipeline
from Toolchain.InstanceStats import InstanceStats
from Toolchain.Solver import Solver
import concurrent.futures
from threading import Event
from typing import Callable, Dict, Optional, Set

def _default_callback(instance: str, data: str):
    logging.info(instance, data)

def _default_err_callback(instance: str, err: str):
    logging.exception(err)

class SingleModelStreamlinerEvaluation:
    def __init__(
        self,
        model: Path,
        working_directory: Path,
        instance_dir: Path,
        training_instances: Set[str],
        training_stats: pd.DataFrame,
        solver: Solver,
        executor: concurrent.futures.ThreadPoolExecutor,
        total_time: Optional[float],
        time_func: Callable[[float], float],
        event: threading.Event,
        streamliner_combination: Optional[str] = None,
    ) -> None:
        self.model = model
        self.working_directory = working_directory
        self.instance_dir = instance_dir
        self.training_instances = training_instances
        self.training_stats = training_stats
        self.solver = solver
        self.executor = executor
        self.total_time: Optional[float] = total_time
        self.time_func = time_func
        self.event = event
        self.streamliner_combination = streamliner_combination

    def generate_pipeline(self, training_instance: str):
        logging.debug(
            f"Generating pipeline for {training_instance} and model {self.model}"
        )

        if self.total_time is None:
            total_time = self.time_func(
                list(
                    self.training_stats[
                        self.training_stats["Instance"] == training_instance
                    ]["TotalTime"]
                )[0]
            )
        else:
            total_time = self.total_time

        return Pipeline(
            self.model,
            self.working_directory,
            self.instance_dir,
            training_instance,
            self.solver,
            self.event,
            total_time,
            self.streamliner_combination,
        )

    def execute(
        self,
        callback: Callable = _default_callback,
        error_callback: Callable = _default_err_callback,
    ) -> Dict[str, InstanceStats]:
        if len(self.training_instances) == 0:
            return {}

        mappings: Dict[str, Pipeline] = {}
        for instance in self.training_instances:
            mappings[instance] = self.generate_pipeline(instance)

        results: Dict[str, InstanceStats] = {}
        
        futures = {
            self.executor.submit(mappings[instance].execute): instance
            for instance in mappings.keys()
        }
        for future in concurrent.futures.as_completed(futures):
            if self.event.is_set():
                break
            
            instance: str = futures[future]
            try:
                data: InstanceStats = future.result()
                results[instance] = data
                # Call the callback method passed in
                callback(instance, data)
            except Exception as exc:
                error_callback(instance, exc)

        return results
